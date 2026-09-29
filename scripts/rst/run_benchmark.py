"""Run methods on benchmarks with Claude (Bedrock) — the part you run on the machine with AWS access.

For every (prompt, seed) the initial program P0 and the self-written requirements are
generated once and cached; every method starts from the same P0 and is charged for
those calls, so comparisons are paired and at equal LLM-call budget. The run resumes:
finished (prompt, seed, method) results are skipped, failed ones are retried.

Examples
    # 1) check Bedrock access and model IDs
    python scripts/check_llm.py
    # 2) tiny smoke test (a few minutes, a few cents)
    python scripts/rst/run_benchmark.py --exp smoke --datasets hardlong --limit 2 --methods zero_shot,rst
    # 3) pilot: 40 CADTestBench prompts, 4 methods, 1 seed
    python scripts/rst/run_benchmark.py --exp pilot --datasets cadtestbench-detailed --limit 40 \
        --methods zero_shot,react,tests_log,rst
    # local end-to-end test without any LLM
    python scripts/rst/run_benchmark.py --exp dry --datasets hardlong --limit 2 --dry-run

Output: runs/<exp>/ (config.json, p0/, reqs/, methods/<method>/<dataset>/<id>_s<seed>.json).
Send back runs/<exp> without the work/ folder (or use scripts/rst/pack_run.py).
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst import roles
from rst.datasets import load
from rst.kernel import Kernel
from rst.llm import LLM, ScriptedLLM, Usage
from rst.pipeline import METHODS, Engine, RSTConfig
from rst.requirements import load_requirements

PRESETS = {
    # name: (base method, RSTConfig overrides)
    "zero_shot": ("zero_shot", {}),
    "react": ("react", {}),
    "tests": ("tests", {}),
    "tests_log": ("tests_log", {}),
    "best_of_n": ("best_of_n", {}),
    "rst": ("rst", {}),
    # ablations
    "rst_llm_localize": ("rst", {"localizer": "llm"}),
    "rst_random_localize": ("rst", {"localizer": "random"}),
    "rst_last_localize": ("rst", {"localizer": "last"}),
    "rst_whole_rewrite": ("rst", {"repair_scope": "whole"}),
    "rst_accept_always": ("rst", {"accept": "always"}),
    "rst_accept_improve": ("rst", {"accept": "improve"}),
    "tests_log_best": ("tests_log", {"return_policy": "best"}),
    # CADSmith with bugs fixed (its own pipeline and budget)
    "cadsmith_fixed": ("cadsmith", {"use_vision": True, "no_leak": False}),
    "cadsmith_fixed_noleak": ("cadsmith", {"use_vision": True, "no_leak": True}),
    "cadsmith_fixed_novision": ("cadsmith", {"use_vision": False, "no_leak": False}),
}
NEEDS_REQS = {"tests", "tests_log", "best_of_n", "rst"}
_print_lock = threading.Lock()


def log(msg, path=None):
    with _print_lock:
        print(msg, flush=True)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def dry_llm() -> ScriptedLLM:
    """Canned answers that exercise every code path without an LLM."""
    def fn(role, system, user):
        if role == "generate":
            return 'import cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10)\nresult = result.faces(">Z").workplane().hole(2)\n'
        if role == "write_requirements":
            return json.dumps({"requirements": [{"id": "R1", "type": "valid"}, {"id": "R2", "type": "single_solid"},
                                                {"id": "R3", "type": "bbox_size", "params": {"x": 12}},
                                                {"id": "R4", "type": "hole_count", "params": {"count": 1, "diameter": 3}}]})
        if role == "repair_local":
            if "S1  <<< EDIT" in user or "[S1]  <<< EDIT" in user:
                return json.dumps({"edits": {"S1": 'result = cq.Workplane("XY").box(12, 10, 10)'}})
            return json.dumps({"edits": {"S2": 'result = result.faces(">Z").workplane().hole(3)'}})
        if role == "refine_whole":
            return 'import cadquery as cq\nresult = cq.Workplane("XY").box(12, 10, 10)\nresult = result.faces(">Z").workplane().hole(3)\n'
        if role == "react":
            return "DONE"
        if role == "fix_error":
            return 'import cadquery as cq\nresult = cq.Workplane("XY").box(10, 10, 10)\n'
        if role == "llm_localize":
            return '{"statement": "S1"}'
        return ""
    return ScriptedLLM(fn)


def make_llm(args):
    return dry_llm() if args.dry_run else LLM(model=args.model, temperature=args.temperature)


def cached_json(path: Path, build):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return data


def run_task(args, exp: Path, entry: dict, seed: int, methods: list[str]):
    ds, eid = entry["dataset"], entry["id"]
    tag = f"{eid}_s{seed}"
    work = exp / "work" / ds / tag
    todo = []
    for m in methods:
        out = exp / "methods" / m / ds / f"{tag}.json"
        if out.exists():
            try:
                if json.loads(out.read_text(encoding="utf-8")).get("status") == "ok":
                    continue
            except Exception:
                pass
        todo.append(m)
    if not todo:
        return "skip"

    # shared P0 (same seed -> same cached program for every method)
    def build_p0():
        llm = make_llm(args)
        code = roles.generate(llm, entry["prompt"], threaded=not args.no_threaded_style)
        u = llm.usage
        return {"code": code, "usage": u.to_dict()}
    p0 = cached_json(exp / "p0" / ds / f"{tag}.json", build_p0)
    pre_p0 = [("generate", p0["usage"]["input_tokens"], p0["usage"]["output_tokens"])]

    reqs = None
    pre_w = []
    if any(PRESETS[m][0] in NEEDS_REQS for m in todo):
        def build_reqs():
            llm = make_llm(args)
            r, errs = roles.write_requirements(llm, entry["prompt"])
            return {"requirements": [x.to_dict() for x in r], "errors": errs, "usage": llm.usage.to_dict()}
        rq = cached_json(exp / "reqs" / ds / f"{tag}.json", build_reqs)
        reqs, _ = load_requirements(rq["requirements"])
        pre_w = [("write_requirements", rq["usage"]["input_tokens"], rq["usage"]["output_tokens"])]

    for m in todo:
        base, overrides = PRESETS[m]
        out = exp / "methods" / m / ds / f"{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            if base == "cadsmith":
                from rst.cadsmith_fixed import run_cadsmith_fixed
                if args.dry_run:
                    raise RuntimeError("cadsmith_fixed cannot run in --dry-run")
                rec = run_cadsmith_fixed(entry["prompt"], tag, str(work / m), use_vision=overrides["use_vision"],
                                         no_leak=overrides["no_leak"], coder_model=args.model,
                                         judge_model=args.cadsmith_judge)
            else:
                cfg = RSTConfig(max_llm_calls=args.budget, seed=seed, **overrides)
                kernel = Kernel(str(work / m), timeout=args.timeout)
                eng = Engine(entry["prompt"], kernel, make_llm(args), tag)
                pre = pre_p0 + (pre_w if base in NEEDS_REQS else [])
                res = METHODS[base](eng, p0["code"], cfg, reqs, pre)
                rec = res.to_dict()
                rec["config"] = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
                if rec.get("final"):  # keep the result small: drop per-row data
                    rec["final"].pop("rows", None)
            rec.update({"status": "ok", "method": m, "dataset": ds, "id": eid, "seed": seed,
                        "model": args.model, "wall_seconds": time.time() - t0})
        except Exception as e:
            rec = {"status": "error", "method": m, "dataset": ds, "id": eid, "seed": seed,
                   "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        out.write_text(json.dumps(rec), encoding="utf-8")
        u = rec.get("usage", {})
        log(f"[{ds} {tag}] {m}: {rec['status']} score={rec.get('final', {}).get('score') if rec.get('final') else None} "
            f"calls={u.get('calls')} stopped={rec.get('stopped')} {rec.get('error', '')}", exp / "progress.log")
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets", default="cadtestbench-detailed")
    ap.add_argument("--methods", default="zero_shot,react,tests_log,rst")
    ap.add_argument("--model", default=os.getenv("RST_MODEL", "claude-sonnet-4-6"))
    ap.add_argument("--cadsmith-judge", default="claude-opus-4-5-20251101")
    ap.add_argument("--budget", type=int, default=10, help="max LLM calls per method per prompt (incl. P0 and writer)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=90, help="seconds per program execution")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-threaded-style", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no LLM: canned answers, for testing the pipeline")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in PRESETS]
    if unknown:
        sys.exit(f"unknown methods {unknown}; choose from {sorted(PRESETS)}")
    seeds = [int(s) for s in args.seeds.split(",")]
    exp = ROOT / "runs" / args.exp
    exp.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        from autofab.llm import describe, resolve_model
        llm_info = {**describe(), "rst_model": resolve_model(args.model)}
    else:
        llm_info = {"backend": "dry-run"}
    cfg_path = exp / "config.json"
    config = {"args": vars(args), "git_commit": git_commit(), "llm": llm_info,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    history = json.loads(cfg_path.read_text(encoding="utf-8")).get("history", []) if cfg_path.exists() else []
    config["history"] = history + [{"started": config["started"], "methods": methods, "git_commit": config["git_commit"]}]
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    tasks = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for e in load(ds, args.limit, args.ids):
            e["dataset"] = ds
            for s in seeds:
                tasks.append((e, s))
    log(f"{len(tasks)} (prompt, seed) tasks x {len(methods)} methods; model={llm_info}", exp / "progress.log")
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_task, args, exp, e, s, methods) for e, s in tasks]
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                log(f"task crashed: {type(e).__name__}: {e}", exp / "progress.log")
            if i % 10 == 0:
                log(f"progress {i}/{len(tasks)} tasks, {time.time() - t0:.0f}s", exp / "progress.log")
    log(f"finished in {time.time() - t0:.0f}s -> {exp}", exp / "progress.log")


if __name__ == "__main__":
    main()
