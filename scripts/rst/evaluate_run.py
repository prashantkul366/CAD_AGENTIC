"""Score a finished run with the held-out evaluators (CPU only, no LLM).

For every method result in runs/<exp>/methods/, the final program is re-executed in a
fresh process and scored with the evaluators that apply to its dataset:

  cadtestbench-*   CADTestBench's own tests: Pass Rate, Requirement Score, Invalid Ratio
  hardlong         the held-out requirement suite shipped with Hard-Long (+ shape metrics)
  cadsmith         shape metrics vs the reference, CADSmith protocol and corrected protocol
  text2cad         Text2CAD-protocol CD x1000 (median / mean / invalid ratio) + corrected metrics

Writes runs/<exp>/eval/records.jsonl, summary.json and summary.md (tables, paired tests,
results by reference complexity).

    python scripts/rst/evaluate_run.py --exp pilot --workers 6
"""

import argparse
import concurrent.futures as cf
import json
import math
import random
import statistics as st
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst.datasets import load
from rst.evaluate import corrected_metrics, run_cadtests, run_dsl_suite, shape_metrics, text2cad_cd
from rst.kernel import Kernel
from rst.thread_program import thread_program

REFCACHE = ROOT / "runs" / "_refcache"
_ref_lock = threading.Lock()


def reference_artifacts(ds: str, entry: dict) -> dict:
    """STEP/STL of the reference and its complexity (modifying statements after threading); cached."""
    d = REFCACHE / ds
    meta_path = d / f"{entry['id']}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    d.mkdir(parents=True, exist_ok=True)
    k = Kernel(str(d), timeout=180, persistent=False)
    res = k.run(entry["reference_code"], [], entry["id"], export=True, summaries=False)
    meta = {"ok": res.success, "stl": res.stl_path, "step": res.step_path}
    try:
        thr, _ = thread_program(entry["reference_code"])
        t = k.run(thr, [], entry["id"] + "_thr", export=False, summaries=False)
        meta["complexity"] = len({r.stmt for r in t.rows if r.stmt is not None}) if t.success else None
    except Exception:
        meta["complexity"] = None
    with _ref_lock:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def evaluate_record(rec: dict, entry: dict, work: Path) -> dict:
    ds = rec["dataset"]
    tag = f"{rec['method']}_{rec['id']}_s{rec['seed']}"
    code = rec.get("final_code")
    out = {k: rec.get(k) for k in ("method", "dataset", "id", "seed")}
    u = rec.get("usage") or {}
    out.update({"llm_calls": u.get("calls"), "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                "kernel_calls": rec.get("kernel_calls"), "seconds": rec.get("wall_seconds"),
                "self_score": (rec.get("final") or {}).get("score")})
    if ds.startswith("cadtestbench-"):
        out["cadtests"] = run_cadtests(code, entry["cadtests"], str(work / "ct"), tag)
        return out
    k = Kernel(str(work / "gen"), timeout=180, persistent=False)
    gen = k.run(code or "", [], tag, export=True, summaries=False) if code else None
    out["invalid"] = not (gen is not None and gen.success)
    if ds == "hardlong":
        out["dsl"] = run_dsl_suite(k, code, entry.get("requirements") or [], tag + "_suite")
    ref = reference_artifacts(ds, entry)
    out["complexity"] = ref.get("complexity")
    if gen is not None and gen.success and ref.get("ok"):
        try:
            if ds == "text2cad":
                out["t2c_cd"] = text2cad_cd(gen.stl_path, ref["stl"])
                out["corrected"] = corrected_metrics(gen.stl_path, ref["stl"], gen.step_path, ref["step"])
            else:
                out.update(shape_metrics(gen.stl_path, ref["stl"], gen.step_path, ref["step"]))
        except Exception as e:
            out["metric_error"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.fmean(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def pass_flag(r) -> bool | None:
    if "cadtests" in r:
        return r["cadtests"]["pass"]
    if "dsl" in r:
        return r["dsl"]["pass"]
    return None


def rs_value(r):
    if "cadtests" in r:
        return r["cadtests"]["rs"]
    if "dsl" in r:
        return r["dsl"]["rs"]
    return None


def invalid_flag(r):
    return r["cadtests"]["invalid"] if "cadtests" in r else r.get("invalid")


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar test on paired pass/fail (a = method, b = comparator)."""
    from scipy.stats import binomtest
    win = sum(1 for x, y in zip(a, b) if x and not y)
    loss = sum(1 for x, y in zip(a, b) if y and not x)
    p = binomtest(win, win + loss, 0.5).pvalue if win + loss else 1.0
    return {"wins": win, "losses": loss, "p": p}


def bootstrap_diff(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    idx = list(range(len(a)))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        diffs.append(sum(a[i] - b[i] for i in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def summarise(records: list[dict]) -> dict:
    by = defaultdict(list)
    for r in records:
        by[(r["dataset"], r["method"])].append(r)
    summary = {}
    for (ds, m), rs in sorted(by.items()):
        row = {"n": len(rs),
               "invalid_ratio": _mean([1.0 if invalid_flag(r) else 0.0 for r in rs]),
               "pass_rate": _mean([1.0 if pass_flag(r) else 0.0 for r in rs]) if pass_flag(rs[0]) is not None else None,
               "requirement_score": _mean([rs_value(r) for r in rs]) if rs_value(rs[0]) is not None else None,
               "llm_calls": _mean([r.get("llm_calls") for r in rs]),
               "input_tokens": _mean([r.get("input_tokens") for r in rs]),
               "output_tokens": _mean([r.get("output_tokens") for r in rs]),
               "kernel_calls": _mean([r.get("kernel_calls") for r in rs])}
        if ds == "text2cad":
            cds = [r.get("t2c_cd") for r in rs if r.get("t2c_cd") is not None]
            row.update({"t2c_cd_median": _median(cds), "t2c_cd_mean": _mean(cds)})
        corr = [r.get("corrected") or {} for r in rs]
        if any(corr):
            row.update({"cd_mm2_median": _median([c.get("cd") for c in corr]), "cd_mm2_mean": _mean([c.get("cd") for c in corr]),
                        "f1_1mm_median": _median([c.get("f1") for c in corr]),
                        "iou_exact_aligned_median": _median([c.get("iou_aligned") for c in corr]),
                        "bbox_err_mm_median": _median([c.get("bbox_err_mm") for c in corr])})
        orig = [r.get("orig") or {} for r in rs]
        if any(orig):
            row.update({"orig_cd_median": _median([o.get("cd") for o in orig]), "orig_cd_mean": _mean([o.get("cd") for o in orig]),
                        "orig_f1_median": _median([o.get("f1") for o in orig]), "orig_iou_median": _median([o.get("iou") for o in orig])})
        summary.setdefault(ds, {})[m] = row
    # paired comparisons against rst and complexity bins
    for ds, methods in summary.items():
        if "rst" not in methods:
            continue
        rst_by = {(r["id"], r["seed"]): r for r in by[(ds, "rst")]}
        for m in methods:
            if m == "rst":
                continue
            other = {(r["id"], r["seed"]): r for r in by[(ds, m)]}
            keys = sorted(set(rst_by) & set(other))
            a = [bool(pass_flag(rst_by[k])) for k in keys]
            b = [bool(pass_flag(other[k])) for k in keys]
            if keys and pass_flag(rst_by[keys[0]]) is not None:
                t = mcnemar(a, b)
                lo, hi = bootstrap_diff([float(x) for x in a], [float(x) for x in b])
                methods[m]["rst_vs_this"] = {"n_paired": len(keys), **t, "pass_rate_diff_ci95": [lo, hi]}
    bins = defaultdict(lambda: defaultdict(list))
    for r in records:
        c = r.get("complexity")
        if c is None or pass_flag(r) is None:
            continue
        b = "1-3" if c <= 3 else "4-7" if c <= 7 else "8+"
        bins[(r["dataset"], b)][r["method"]].append(1.0 if pass_flag(r) else 0.0)
    summary["_by_complexity"] = {f"{ds}|{b}": {m: {"n": len(v), "pass_rate": _mean(v)} for m, v in ms.items()}
                                 for (ds, b), ms in sorted(bins.items())}
    return summary


def to_markdown(summary: dict) -> str:
    lines = []
    for ds, methods in summary.items():
        if ds.startswith("_"):
            continue
        lines.append(f"## {ds}\n")
        cols = ["n", "pass_rate", "requirement_score", "invalid_ratio", "t2c_cd_median", "t2c_cd_mean", "cd_mm2_median",
                "f1_1mm_median", "iou_exact_aligned_median", "orig_cd_mean", "llm_calls", "input_tokens", "output_tokens",
                "kernel_calls"]
        cols = [c for c in cols if any(methods[m].get(c) is not None for m in methods)]
        lines.append("| method | " + " | ".join(cols) + " | vs RST (wins/losses, p) |")
        lines.append("|---" * (len(cols) + 2) + "|")
        for m, row in methods.items():
            cells = []
            for c in cols:
                v = row.get(c)
                cells.append("—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
            cmp = row.get("rst_vs_this")
            cmp_s = f"{cmp['wins']}/{cmp['losses']}, p={cmp['p']:.3g}" if cmp else ""
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {cmp_s} |")
        lines.append("")
    if summary.get("_by_complexity"):
        lines.append("## Pass rate by reference complexity (modifying statements)\n")
        for key, ms in summary["_by_complexity"].items():
            lines.append(f"- {key}: " + ", ".join(f"{m} {v['pass_rate']:.2f} (n={v['n']})" for m, v in sorted(ms.items())))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--methods", default=None, help="comma list; default all in the run")
    args = ap.parse_args()
    exp = ROOT / "runs" / args.exp
    out_dir = exp / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    done = {}
    if rec_path.exists():
        for line in open(rec_path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["method"], r["dataset"], r["id"], r["seed"])] = r
    entries = {}
    jobs = []
    wanted = set(args.methods.split(",")) if args.methods else None
    for f in sorted((exp / "methods").glob("*/*/*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") != "ok" or (wanted and rec["method"] not in wanted):
            continue
        key = (rec["method"], rec["dataset"], rec["id"], rec["seed"])
        if key in done:
            continue
        ds = rec["dataset"]
        if ds not in entries:
            entries[ds] = {e["id"]: e for e in load(ds)}
        jobs.append((rec, entries[ds][rec["id"]]))
    print(f"{len(jobs)} results to evaluate ({len(done)} already done)", flush=True)
    lock = threading.Lock()

    def one(job):
        rec, entry = job
        r = evaluate_record(rec, entry, exp / "eval_work")
        with lock:
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        return r

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(one, jobs), 1):
            if i % 20 == 0:
                print(f"evaluated {i}/{len(jobs)}", flush=True)
    records = [json.loads(l) for l in open(rec_path, encoding="utf-8")] if rec_path.exists() else []
    summary = summarise(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
