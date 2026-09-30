"""E3: does the satisfaction matrix find the broken operation?

Inject single-statement faults into reference programs and ask each
localiser for the culprit statement:

  matrix   RST blame rules (regression / last-touch / missing)
  llm      the LLM, shown the program, failing requirements and final measurements  (--llm; needs Bedrock)
  random   a random statement among those that changed the solid
  last     the last statement that changed the solid

Specification source (--spec):
  oracle   derived from the reference solid (no LLM; an upper-bound condition)
  entry    the dataset's own requirement suite (Hard-Long ships one)
  self     written by the LLM from the prompt only (needs Bedrock)

Gate G1 (research plan): matrix top-1 >= 70% and >= 15 points above the LLM.

    python scripts/rst/e3_localization.py --dataset cadsmith --spec oracle --threaded
"""

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rst.datasets import load
from rst.kernel import Kernel
from rst.localize import last_statement_localize, llm_localize, localize, modifying_statements, random_localize
from rst.matrix import Trajectory
from rst.mutants import make_mutants, normalised
from rst.program import Program
from rst.requirements import load_requirements
from rst.thread_program import thread_program


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cadsmith")
    ap.add_argument("--spec", choices=["oracle", "entry", "self"], default="oracle")
    ap.add_argument("--threaded", action="store_true", help="convert references to one-feature-per-statement form first")
    ap.add_argument("--n-mutants", type=int, default=5)
    ap.add_argument("--min-statements", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="also run the LLM localiser (Bedrock)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/e3")
    args = ap.parse_args()

    tag = f"{args.dataset}_{args.spec}{'_thr' if args.threaded else ''}{'_llm' if args.llm else ''}_s{args.seed}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel = Kernel(str(out_dir / "work"), timeout=120)
    llm = None
    if args.llm or args.spec == "self":
        from rst.llm import LLM
        llm = LLM(model=args.model, temperature=0.0)
    rng = random.Random(args.seed)
    records = []
    skipped = defaultdict(int)

    for e in load(args.dataset, args.limit):
        code = normalised(e["reference_code"])
        if args.threaded:
            code, _ = thread_program(code)
            code = normalised(code)
        base = kernel.run(code, [], e["id"] + "_ref", export=False, autospec=(args.spec == "oracle"))
        if not base.success:
            skipped["reference_failed"] += 1
            continue
        if args.spec == "oracle":
            spec_raw = base.autospec or []
        elif args.spec == "entry":
            spec_raw = e.get("requirements") or []
        else:
            from rst.roles import write_requirements
            reqs_self, _ = write_requirements(llm, e["prompt"])
            spec_raw = [r.to_dict() for r in reqs_self]
        reqs, _ = load_requirements(spec_raw)
        ref_eval = kernel.run(code, reqs, e["id"] + "_refeval", export=False)
        # keep only requirements the reference satisfies (a failing one cannot localise anything)
        keep = [r for r, ok in zip(reqs, ref_eval.final_pass()) if ok] if ref_eval.success else []
        if not keep:
            skipped["empty_spec"] += 1
            continue
        if len(modifying_statements(Trajectory(ref_eval))) < args.min_statements:
            skipped["too_few_statements"] += 1
            continue
        extent = max(ref_eval.geometry["bounding_box"][k] for k in ("xlen", "ylen", "zlen"))
        ref_modifying = set(modifying_statements(Trajectory(ref_eval)))
        prog = Program(code)
        for m in make_mutants(code, args.n_mutants, seed=args.seed, extent=extent):
            res = kernel.run(m.code, keep, f"{e['id']}_m{len(records)}", export=False)
            if not res.success:
                skipped["mutant_exec_failed"] += 1
                continue
            if res.all_pass():
                skipped["mutant_equivalent_under_spec"] += 1
                continue
            traj = Trajectory(res)
            blames = localize(traj)
            prog_m = Program(m.code)
            is_delete = m.kind == "feature_delete"

            def hit(stmt, insert=False):
                """Region hit: the injected statement is inside the repair region of the blamed one
                (blamed statement + its definitions). Deleted features: a blame next to the gap, or an insertion."""
                if is_delete:
                    return bool(insert) or (stmt is not None and abs(stmt - m.stmt) <= 1)
                if stmt is None:
                    return False
                return m.stmt == stmt or m.stmt in prog_m.dependencies(stmt, 2)

            top = blames[0] if blames else None
            rnd = random_localize(traj, rng)
            last = last_statement_localize(traj)
            rec = {
                "id": e["id"], "kind": m.kind, "stmt": m.stmt, "detail": m.detail,
                "param_mutant": m.stmt not in ref_modifying and not is_delete,
                "n_modifying": len(modifying_statements(traj)),
                "matrix": top.stmt if top else None, "matrix_rule": top.rule if top else None,
                "matrix_strict": top is not None and top.stmt == m.stmt,
                "matrix_hit": top is not None and hit(top.stmt, top.insert),
                "matrix_top3": any(hit(b.stmt, b.insert) for b in blames[:3]),
                "random_hit": hit(rnd), "last_hit": hit(last),
                "n_failing": len(traj.failing()),
            }
            if args.llm:
                pick = llm_localize(llm, e["prompt"], prog_m, traj)
                rec["llm"] = pick
                rec["llm_strict"] = pick == m.stmt
                rec["llm_hit"] = hit(pick)
            records.append(rec)
            print(json.dumps(rec))

    def rate(key, rows):
        rows = [r for r in rows if key in r]
        return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else None

    from rst.localize import RULES_VERSION
    summary = {"tag": tag, "rules": RULES_VERSION, "n_mutants": len(records), "skipped": dict(skipped)}
    (out_dir / "RULES").write_text(RULES_VERSION, encoding="utf-8")
    for key in ("matrix_hit", "matrix_strict", "matrix_top3", "llm_hit", "llm_strict", "random_hit", "last_hit"):
        summary[key] = rate(key, records)
    for flag, name in ((True, "parameter_mutants"), (False, "operation_mutants")):
        sub = [r for r in records if r.get("param_mutant") == flag]
        summary[name] = {"n": len(sub), **{k: rate(k, sub) for k in ("matrix_hit", "llm_hit", "random_hit", "last_hit")}}
    summary["by_kind"] = {k: {key: rate(key, [r for r in records if r["kind"] == k])
                              for key in ("matrix_hit", "llm_hit", "random_hit", "last_hit")}
                          for k in sorted({r["kind"] for r in records})}
    summary["by_rule"] = {k: {"n": sum(r["matrix_rule"] == k for r in records),
                              "matrix_hit": rate("matrix_hit", [r for r in records if r["matrix_rule"] == k])}
                          for k in sorted({str(r["matrix_rule"]) for r in records})}
    if summary.get("llm_hit") is not None and summary.get("matrix_hit") is not None:
        summary["gate_G1"] = summary["matrix_hit"] >= 0.70 and summary["matrix_hit"] - summary["llm_hit"] >= 0.15
    if llm is not None:
        summary["llm_usage"] = llm.usage.to_dict()
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
