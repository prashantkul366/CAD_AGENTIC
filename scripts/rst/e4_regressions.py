"""E4: do whole-program refinements break requirements that already held?

For each refinement round (program before -> program after), evaluate both with a
held-out suite and count rounds in which at least one previously satisfied
requirement fails afterwards.

Sources:
  --cadsmith-results FILE   a CADSmith results.jsonl (per-iteration code is stored there);
                            held-out suite = oracle requirements derived from the reference solid
  --exp NAME                one of our runs (rounds store their code); held-out suite = the
                            dataset's own tests (CADTestBench) or requirement suite (Hard-Long)

Gate (research plan): regressions in >= 15% of rounds, or >= 50% of failures locally fixable.

    python scripts/rst/e4_regressions.py --cadsmith-results reprod/results.jsonl --dataset cadsmith
"""

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst.datasets import load
from rst.evaluate import run_cadtests
from rst.kernel import Kernel
from rst.requirements import load_requirements


def held_out_passes(ds, entry, code, kernel, work, tag, suite_cache):
    """Per-requirement pass list for one program (None if it does not run)."""
    if ds.startswith("cadtestbench-"):
        r = run_cadtests(code, entry["cadtests"], work, tag)
        if r["invalid"]:
            return None
        # per-test detail is not returned by run_cadtests; use requirement groups via rs * groups
        return [r["tests_passed"], r["tests_total"]]
    suite = suite_cache.get(entry["id"])
    if suite is None:
        if ds == "hardlong":
            suite, _ = load_requirements(entry["requirements"])
        else:
            ref = kernel.run(entry["reference_code"], [], tag + "_ref", export=False, autospec=True)
            suite, _ = load_requirements(ref.autospec or [])
        suite_cache[entry["id"]] = suite
    res = kernel.run(code, suite, tag, export=False, summaries=False)
    return res.final_pass() if res.success else None


def rounds_from_cadsmith(path: Path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        its = r.get("per_iteration") or []
        codes = [it.get("code") for it in its if it.get("code")]
        for a, b in zip(codes, codes[1:]):
            yield r["id"], a, b


def rounds_from_run(exp: Path, methods: list[str]):
    for f in sorted((exp / "methods").glob("*/*/*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") != "ok" or rec["method"] not in methods:
            continue
        p0 = json.loads((exp / "p0" / rec["dataset"] / f"{rec['id']}_s{rec['seed']}.json").read_text(encoding="utf-8"))["code"]
        prev = p0
        for rd in rec.get("rounds", []):
            code = rd.get("code")
            if not code or rd.get("result") == "exec_failed" or rd.get("success") is False:
                continue
            yield (rec["method"], rec["dataset"], rec["id"]), prev, code
            if rd.get("accepted", True):   # whole-program methods continue from the latest program
                prev = code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadsmith-results", default=None)
    ap.add_argument("--dataset", default="cadsmith")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--methods", default="tests_log,react,rst")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/e4")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kernel = Kernel(str(out / "work"), timeout=120)
    suite_cache = {}
    jobs = []
    if args.cadsmith_results:
        entries = {e["id"]: e for e in load(args.dataset)}
        for i, (eid, a, b) in enumerate(rounds_from_cadsmith(Path(args.cadsmith_results))):
            jobs.append((("cadsmith_run", args.dataset, eid), entries[eid], a, b, i))
    if args.exp:
        exp = ROOT / "runs" / args.exp
        cache = {}
        for i, (key, a, b) in enumerate(rounds_from_run(exp, args.methods.split(","))):
            ds = key[1]
            if ds not in cache:
                cache[ds] = {e["id"]: e for e in load(ds)}
            jobs.append((key, cache[ds][key[2]], a, b, i))
    print(len(jobs), "refinement rounds", flush=True)

    def one(job):
        key, entry, a, b, i = job
        ds = key[1]
        pa = held_out_passes(ds, entry, a, kernel, str(out / "ct"), f"r{i}a", suite_cache)
        pb = held_out_passes(ds, entry, b, kernel, str(out / "ct"), f"r{i}b", suite_cache)
        rec = {"method": key[0], "dataset": ds, "id": key[2], "before_ok": pa is not None, "after_ok": pb is not None}
        if pa is not None and pb is not None:
            if ds.startswith("cadtestbench-"):
                rec["regressed"] = pb[0] < pa[0]
                rec["delta_tests"] = pb[0] - pa[0]
            else:
                broken = [k for k, (x, y) in enumerate(zip(pa, pb)) if x and not y]
                fixed = [k for k, (x, y) in enumerate(zip(pa, pb)) if y and not x]
                rec.update({"regressed": bool(broken), "n_broken": len(broken), "n_fixed": len(fixed)})
        return rec

    with cf.ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, jobs))
    summary = {}
    for m in sorted({r["method"] for r in recs}):
        rs = [r for r in recs if r["method"] == m and "regressed" in r]
        summary[m] = {"rounds": len(rs), "regression_rate": (sum(r["regressed"] for r in rs) / len(rs)) if rs else None,
                      "rounds_breaking_something_they_also_fix": sum(1 for r in rs if r.get("n_broken") and r.get("n_fixed"))}
    summary["gate_E4_regressions_ge_15pct"] = {m: (v["regression_rate"] or 0) >= 0.15 for m, v in summary.items()
                                               if isinstance(v, dict)}
    (out / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
