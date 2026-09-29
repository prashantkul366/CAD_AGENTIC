"""E1: can reference programs be traced state by state, and converted to one-feature-per-statement form?

For every reference program: trace it as written, rewrite it into
state-threaded form, trace that too, and check the rewrite builds the same
solid (volume and bounding box). No LLM needed.

    python scripts/rst/e1_trace_coverage.py --dataset cadsmith
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rst.datasets import load
from rst.kernel import Kernel
from rst.thread_program import thread_program


def n_modifying(res):
    return len({r.stmt for r in res.rows if r.stmt is not None}) if res.success else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cadsmith")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/e1")
    args = ap.parse_args()
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel = Kernel(str(out_dir / "work"), timeout=120)
    rows = []
    for e in load(args.dataset, args.limit):
        code = e["reference_code"]
        try:
            threaded, added = thread_program(code)
        except SyntaxError as ex:
            threaded, added = None, 0
        a = kernel.run(code, [], e["id"] + "_orig", export=False)
        b = kernel.run(threaded, [], e["id"] + "_thr", export=False) if threaded else None
        same = None
        if a.success and b is not None and b.success:
            va, vb = a.geometry["volume"], b.geometry["volume"]
            ba, bb = a.geometry["bounding_box"], b.geometry["bounding_box"]
            same = abs(va - vb) <= 1e-6 * max(abs(va), 1.0) and all(abs(ba[k] - bb[k]) < 1e-6 for k in ("xlen", "ylen", "zlen"))
        rows.append({
            "id": e["id"], "tier": e.get("tier"), "exec": a.success, "error_type": a.error_type,
            "rows": len(a.rows) if a.success else 0, "stmts_modifying": n_modifying(a),
            "lineage": bool(a.coverage.get("lineage")) if a.success else False,
            "threaded_exec": bool(b and b.success), "threaded_same": same, "statements_added": added,
            "threaded_stmts_modifying": n_modifying(b) if b else 0,
        })
        print(json.dumps(rows[-1]))
    ok = [r for r in rows if r["exec"]]
    summ = {
        "dataset": args.dataset, "n": len(rows),
        "exec_rate": len(ok) / max(len(rows), 1),
        "lineage_rate": sum(r["lineage"] for r in ok) / max(len(ok), 1),
        "threaded_equivalent_rate": sum(bool(r["threaded_same"]) for r in ok) / max(len(ok), 1),
        "mean_states": st.fmean([r["rows"] for r in ok]) if ok else 0,
        "mean_modifying_statements_before": st.fmean([r["stmts_modifying"] for r in ok]) if ok else 0,
        "mean_modifying_statements_after": st.fmean([r["threaded_stmts_modifying"] for r in ok]) if ok else 0,
        "share_ge3_modifying_before": sum(r["stmts_modifying"] >= 3 for r in ok) / max(len(ok), 1),
        "share_ge3_modifying_after": sum(r["threaded_stmts_modifying"] >= 3 for r in ok) / max(len(ok), 1),
        "gate_G0_convert_and_execute_ge_0.70": (sum(bool(r["threaded_same"]) for r in rows) / max(len(rows), 1)) >= 0.70,
    }
    (out_dir / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
