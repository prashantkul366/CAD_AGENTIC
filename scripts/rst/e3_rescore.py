"""Re-score finished E3 runs from their saved kernel outputs, without executing any CAD code.

Every E3 fault leaves its program, requirements and full trajectory in runs/e3/<tag>/work/<id>_m<k>.*
(k = index of the record). This rebuilds each trajectory and applies a blame-rule version and the
spectrum-based baselines to it, so rule versions are compared on exactly the same faults.
The random baseline replays the original random stream; every localiser is scored with the same repair region.

    python scripts/rst/e3_rescore.py --rules v2 --check      # must reproduce the stored matrix columns
    python scripts/rst/e3_rescore.py --rules v3              # writes runs/e3_rescored/v3/<tag>/records.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst.kernel import KernelResult, Row
from rst.localize import (RULES, SBFL_FORMULAS, last_statement_localize, localize, random_localize, region_chains,
                          sbfl_rank)
from rst.matrix import Trajectory
from rst.program import Program
from rst.requirements import load_requirements


def load_result(work: Path, name: str) -> KernelResult:
    job = json.loads((work / f"{name}.job.json").read_text(encoding="utf-8"))
    data = json.loads((work / f"{name}.out.json").read_text(encoding="utf-8"))
    code = (work / f"{name}.py").read_text(encoding="utf-8")
    reqs, _ = load_requirements(job["requirements"])
    rows = [Row(i, r.get("stmt"), r.get("op", "?"), r.get("lineno"), r.get("verdicts", []), r.get("summary"))
            for i, r in enumerate(data["rows"])]
    return KernelResult(success=True, code=code, requirements=reqs, rows=rows, geometry=data.get("geometry"),
                        final_summary=data.get("final_summary"))


def rescore(run: Path, rules: str, region: str) -> list:
    seed = int(run.name.rsplit("_s", 1)[1])
    rng = random.Random(seed)                   # same streams as e3_localization.py
    rng_sbfl = random.Random(seed + 1000)
    old = [json.loads(l) for l in open(run / "records.jsonl", encoding="utf-8") if l.strip()]
    out = []
    for k, rec in enumerate(old):
        res = load_result(run / "work", f"{rec['id']}_m{k}")
        traj = Trajectory(res)
        prog = Program(res.code)
        is_delete = rec["kind"] == "feature_delete"

        def hit(stmt, insert=False):
            if is_delete:
                return bool(insert) or (stmt is not None and abs(stmt - rec["stmt"]) <= 1)
            if stmt is None:
                return False
            return rec["stmt"] == stmt or rec["stmt"] in prog.dependencies(stmt, 2, chains=region_chains(region))

        blames = localize(traj, rules)
        top = blames[0] if blames else None
        new = {k2: rec[k2] for k2 in ("id", "kind", "stmt", "detail", "param_mutant", "n_modifying") if k2 in rec}
        new["random_hit"] = hit(random_localize(traj, rng))       # replayed, so it is scored with the same region
        new["last_hit"] = hit(last_statement_localize(traj))
        new.update({"matrix": top.stmt if top else None, "matrix_rule": top.rule if top else None,
                    "matrix_strict": top is not None and top.stmt == rec["stmt"],
                    "matrix_hit": top is not None and hit(top.stmt, top.insert),
                    "matrix_top3": any(hit(b.stmt, b.insert) for b in blames[:3]),
                    "n_failing": len(traj.failing())})
        for name in SBFL_FORMULAS:
            ranked, tied = sbfl_rank(traj, name, rng_sbfl)
            new[f"{name}_hit"] = bool(ranked) and hit(ranked[0])
            new[f"{name}_top3"] = any(hit(s) for s in ranked[:3])
            new[f"{name}_exp"] = sum(hit(s) for s in tied) / len(tied) if tied else 0.0
        if "llm" in rec:
            new.update({"llm": rec["llm"], "llm_strict": rec["llm"] == rec["stmt"], "llm_hit": hit(rec["llm"])})
        out.append(new)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, choices=RULES)
    ap.add_argument("--runs", default="runs/e3")
    ap.add_argument("--region", default=None, choices=RULES, help="repair region used to score a hit (default: as --rules)")
    ap.add_argument("--out", default="runs/e3_rescored")
    ap.add_argument("--check", action="store_true", help="compare the matrix columns with the stored records")
    args = ap.parse_args()
    for run in sorted((ROOT / args.runs).glob("*_s*")):
        if not (run / "records.jsonl").exists():
            continue
        stored_rules = (run / "RULES").read_text(encoding="utf-8").strip() if (run / "RULES").exists() else "v1"
        new = rescore(run, args.rules, args.region or args.rules)
        dest = ROOT / args.out / (args.rules + (f"_region{args.region}" if args.region else "")) / run.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "records.jsonl").write_text("\n".join(json.dumps(r) for r in new) + "\n", encoding="utf-8")
        (dest / "RULES").write_text(args.rules, encoding="utf-8")
        line = f"{run.name}: {len(new)} faults, matrix first {sum(r['matrix_hit'] for r in new) / len(new):.3f}"
        if args.check and stored_rules == args.rules:
            old = [json.loads(l) for l in open(run / "records.jsonl", encoding="utf-8") if l.strip()]
            diff = sum(any(a[c] != b[c] for c in ("matrix", "matrix_hit", "matrix_top3", "random_hit", "last_hit"))
                       for a, b in zip(old, new))
            line += f"  | differs from stored {stored_rules} records: {diff}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
