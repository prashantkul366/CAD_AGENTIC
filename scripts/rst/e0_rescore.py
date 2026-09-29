"""E0: re-score the CADSmith reproduction under the original and the corrected protocol.

The final programs are stored in the results files, so parts are rebuilt here
(CadQuery is deterministic) together with the references; no STL files are needed.

    python scripts/rst/e0_rescore.py --run full_vision=reprod/results.jsonl \
        --run no_vision="reprod/results (1).jsonl" --run zeroshot="reprod/results (2).jsonl"
"""

import argparse
import concurrent.futures as cf
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst.datasets import load
from rst.evaluate import shape_metrics
from rst.kernel import Kernel


def final_code(rec: dict):
    if "per_iteration" in rec:
        ok = [it for it in rec["per_iteration"] if it.get("execution_success") and it.get("code")]
        return ok[-1]["code"] if ok else None
    return rec.get("code") if rec.get("execution_success") else None


def agg(vals, fn):
    vals = [v for v in vals if v is not None]
    return round(fn(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path/to/results.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/e0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = {e["id"]: e for e in load("cadsmith")}
    ref_kernel = Kernel(str(out / "refs"), timeout=180, persistent=False)
    refs = {}

    def ref(eid):
        if eid not in refs:
            r = ref_kernel.run(entries[eid]["reference_code"], [], eid, export=True, summaries=False)
            refs[eid] = r if r.success else None
        return refs[eid]

    # build references first (sequential, cached)
    for eid in entries:
        ref(eid)
    summary = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        k = Kernel(str(out / name), timeout=180, persistent=False)

        def one(rec):
            code = final_code(rec)
            row = {"id": rec["id"], "tier": rec.get("tier"), "reported": rec.get("metrics")}
            r = ref(rec["id"])
            if not code or r is None:
                row["invalid"] = True
                return row
            g = k.run(code, [], rec["id"], export=True, summaries=False)
            if not g.success:
                row["invalid"] = True
                return row
            row["invalid"] = False
            row.update(shape_metrics(g.stl_path, r.stl_path, g.step_path, r.step_path))
            return row

        with cf.ThreadPoolExecutor(args.workers) as ex:
            rows = list(ex.map(one, recs))
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        ok = [r for r in rows if not r["invalid"]]
        s = {"n": len(rows), "exec_rate": len(ok) / len(rows)}
        for proto in ("orig", "corrected"):
            vals = [r.get(proto) or {} for r in ok]
            s[proto] = {"cd_median": agg([v.get("cd") for v in vals], st.median),
                        "cd_mean": agg([v.get("cd") for v in vals], st.fmean),
                        "f1_median": agg([v.get("f1") for v in vals], st.median)}
            if proto == "orig":
                s[proto]["iou_median"] = agg([v.get("iou") for v in vals], st.median)
            else:
                s[proto].update({
                    "iou_exact_as_placed_median": agg([v.get("iou_as_placed") for v in vals], st.median),
                    "iou_exact_aligned_median": agg([v.get("iou_aligned") for v in vals], st.median),
                    "bbox_err_mm_median": agg([v.get("bbox_err_mm") for v in vals], st.median),
                    "bbox_err_gt_1mm": sum(1 for v in vals if (v.get("bbox_err_mm") or 0) > 1.0),
                    "volume_err_gt_5pct": sum(1 for v in vals if (v.get("volume_err_pct") or 0) > 5.0),
                    "cd_noise_floor_median": agg([v.get("cd_noise_floor") for v in vals], st.median),
                })
        reported = [r["reported"] for r in rows if r.get("reported")]
        s["reported"] = {"cd_median": agg([m["chamfer_distance"] for m in reported], st.median),
                         "cd_mean": agg([m["chamfer_distance"] for m in reported], st.fmean),
                         "f1_median": agg([m["f1_score"] for m in reported], st.median),
                         "iou_median": agg([m["volumetric_iou"] for m in reported], st.median)}
        summary[name] = s
        print(name, json.dumps(s, indent=1), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
