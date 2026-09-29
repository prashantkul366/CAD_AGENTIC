"""Check our CADTestBench evaluator against the paper before trusting it.

1. Reference programs should pass (almost) all of their own tests.
2. Re-scoring the released Claude-4.6-Sonnet outputs should reproduce
   Table 3 of arXiv 2605.07807 (PR / RS / IR).

    python scripts/data/fetch_cadtestbench_baselines.py --models Claude-4.6-Sonnet
    python scripts/rst/validate_cadtests_eval.py --workers 6
"""

import argparse
import concurrent.futures as cf
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rst.datasets import EXTERNAL, load
from rst.evaluate import run_cadtests

PAPER = {  # Table 3, Claude-4.6-Sonnet: (IR, RS, PR)
    ("plain", "detailed"): ("ReAct", 0.0, 0.874, 0.580), ("plain", "abstract"): ("ReAct", 0.0, 0.929, 0.715),
    ("CADTests", "detailed"): ("CADTests", 0.005, 0.882, 0.590), ("CADTests", "abstract"): ("CADTests", 0.0, 0.953, 0.765),
    ("CADTests_Log", "detailed"): ("CADTests+Log", 0.0, 0.897, 0.625),
    ("CADTests_Log", "abstract"): ("CADTests+Log", 0.0, 0.962, 0.810),
}


def score(rows, codes, workdir, workers):
    def one(r):
        return run_cadtests(codes.get(r["id"]), r["cadtests"], workdir, r["id"])
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "records.jsonl").write_text(
        "\n".join(json.dumps({"id": r["id"], **x}) for r, x in zip(rows, res)) + "\n", encoding="utf-8")
    return {"PR": st.fmean(x["pass"] for x in res), "RS": st.fmean(x["rs"] for x in res),
            "IR": st.fmean(x["invalid"] for x in res), "n": len(res),
            "test_pass_rate": sum(x["tests_passed"] for x in res) / max(sum(x["tests_total"] for x in res), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="Claude-4.6-Sonnet")
    ap.add_argument("--out", default="runs/validate_cadtests")
    ap.add_argument("--methods", default="plain,CADTests,CADTests_Log")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()
    out = {}
    for part in ("detailed", "abstract"):
        rows = load(f"cadtestbench-{part}")
        if not args.skip_reference:
            out[f"reference/{part}"] = score(rows, {r["id"]: r["reference_code"] for r in rows},
                                             f"{args.out}/ref_{part}", args.workers)
            print(part, "reference", out[f"reference/{part}"], flush=True)
        for method in args.methods.split(","):
            d = EXTERNAL / "cadtestbench" / "baselines" / method / args.model / part
            if not d.exists():
                continue
            codes = {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in d.glob("*.py")}
            s = score(rows, codes, f"{args.out}/{method}_{part}", args.workers)
            name, ir, rs, pr = PAPER[(method, part)]
            s["paper"] = {"method": name, "IR": ir, "RS": rs, "PR": pr}
            out[f"{method}/{part}"] = s
            print(part, method, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}, flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps({**old, **out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
