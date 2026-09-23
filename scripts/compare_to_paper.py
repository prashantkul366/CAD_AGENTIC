"""Compare reproduction results with the numbers reported in the CADSmith paper.

Reads results/<name>/results.jsonl for the three configurations and prints
Table I, Table II, the vision ablation, convergence statistics and the
per-entry cases the paper discusses, each next to the paper's value. The
same report is written to results/paper_comparison.md.

Usage:
    python scripts/compare_to_paper.py
    python scripts/compare_to_paper.py --full full_vision --no-vision no_vision --zeroshot zeroshot
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Paper values (arXiv:2603.26512).
PAPER_TABLE1 = {  # Exec%, CD med, CD mean, F1 med, IoU med
    "Zero-shot": (95, 0.55, 28.37, 0.9707, 0.8085),
    "No-vision": (99, 0.48, 18.19, 0.9792, 0.9563),
    "Full (vision)": (100, 0.48, 0.74, 0.9846, 0.9629),
}
PAPER_TABLE2 = {  # N, CD med, CD mean, F1 med, IoU med
    "T1": (50, 0.32, 0.47, 0.9985, 0.9834),
    "T2": (25, 0.32, 0.58, 0.9979, 0.7661),
    "T3": (25, 0.96, 1.42, 0.8859, 0.9582),
}
PAPER_T3_ABLATION = {  # T3 mean CD, T3 mean F1
    "No-vision": (49.68, 0.74),
    "Full (vision)": (1.42, 0.85),
}
PAPER_CONVERGENCE = (88, 0.13)  # converged at iteration 0, mean refinement iterations
PAPER_IMPROVEMENTS = {"T3_023": (0.037, 0.943), "T1_021": (0.168, 0.995), "T2_005": (0.095, 0.867)}
PAPER_FAILURES = {"T3_016": 0.57, "T3_024": 0.59, "T3_019": 0.963}


def load(name):
    path = RESULTS_DIR / name / "results.jsonl"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping")
        return None
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[r["id"]] = r  # a re-run of an entry replaces the earlier record
    return list(records.values())


def metric(records, key):
    return [r["metrics"][key] for r in records if r.get("metrics")]


def summary(records):
    cd, f1, iou = (metric(records, k) for k in ("chamfer_distance", "f1_score", "volumetric_iou"))
    n = len(records)
    exec_pct = 100 * sum(bool(r.get("execution_success")) for r in records) / n if n else float("nan")
    med = lambda v: statistics.median(v) if v else float("nan")
    mean = lambda v: statistics.fmean(v) if v else float("nan")
    return {"n": n, "n_metrics": len(cd), "exec": exec_pct, "cd_med": med(cd), "cd_mean": mean(cd),
            "f1_med": med(f1), "f1_mean": mean(f1), "iou_med": med(iou)}


def fmt(x, digits):
    return "—" if x is None or x != x else f"{x:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", default="full_vision")
    parser.add_argument("--no-vision", default="no_vision")
    parser.add_argument("--zeroshot", default="zeroshot")
    args = parser.parse_args()

    runs = {"Zero-shot": load(args.zeroshot), "No-vision": load(args.no_vision), "Full (vision)": load(args.full)}
    out = []
    p = out.append

    p("# CADSmith reproduction vs paper\n")
    for label, name in (("Zero-shot", args.zeroshot), ("No-vision", args.no_vision), ("Full (vision)", args.full)):
        cfg = RESULTS_DIR / name / "config.json"
        if cfg.exists():
            llm = json.loads(cfg.read_text(encoding="utf-8")).get("llm", {})
            p(f"- {label} (`{name}`): coder `{llm.get('coder_model', '?')}`, judge `{llm.get('judge_model', '?')}`")
    p("")

    p("## Table I: overall (100 entries)\n")
    p("| Configuration | Source | Exec % | CD med | CD mean | F1 med | IoU med |")
    p("|---|---|---|---|---|---|---|")
    for label, records in runs.items():
        e, cdm, cdmean, f1m, ioum = PAPER_TABLE1[label]
        p(f"| {label} | paper | {e} | {cdm:.2f} | {cdmean:.2f} | {f1m:.4f} | {ioum:.4f} |")
        if records:
            s = summary(records)
            p(f"| {label} | **ours** (n={s['n']}) | {fmt(s['exec'], 0)} | {fmt(s['cd_med'], 2)} | "
              f"{fmt(s['cd_mean'], 2)} | {fmt(s['f1_med'], 4)} | {fmt(s['iou_med'], 4)} |")
    p("")

    full = runs["Full (vision)"]
    if full:
        p("## Table II: full pipeline by tier\n")
        p("| Tier | Source | N | CD med | CD mean | F1 med | IoU med |")
        p("|---|---|---|---|---|---|---|")
        for tier, (n, cdm, cdmean, f1m, ioum) in PAPER_TABLE2.items():
            p(f"| {tier} | paper | {n} | {cdm:.2f} | {cdmean:.2f} | {f1m:.4f} | {ioum:.4f} |")
            s = summary([r for r in full if r.get("tier") == tier])
            p(f"| {tier} | **ours** | {s['n_metrics']} | {fmt(s['cd_med'], 2)} | {fmt(s['cd_mean'], 2)} | "
              f"{fmt(s['f1_med'], 4)} | {fmt(s['iou_med'], 4)} |")
        p("")

    p("## Vision ablation on T3\n")
    p("| Configuration | T3 CD mean (paper → ours) | T3 F1 mean (paper → ours) |")
    p("|---|---|---|")
    for label, (cd_p, f1_p) in PAPER_T3_ABLATION.items():
        records = runs[label]
        s = summary([r for r in records if r.get("tier") == "T3"]) if records else None
        p(f"| {label} | {cd_p:.2f} → {fmt(s and s['cd_mean'], 2)} | {f1_p:.2f} → {fmt(s and s['f1_mean'], 2)} |")
    p("")

    if full:
        first = sum(1 for r in full if r.get("converged") and r.get("num_iterations") == 1)
        refine = [max(r.get("num_iterations", 1) - 1, 0) for r in full if "num_iterations" in r]
        mean_refine = statistics.fmean(refine) if refine else float("nan")
        p("## Convergence (full pipeline)\n")
        p(f"- Converged at iteration 0: paper {PAPER_CONVERGENCE[0]}/100 → ours {first}/{len(full)}")
        p(f"- Mean refinement iterations per entry: paper {PAPER_CONVERGENCE[1]:.2f} → ours {mean_refine:.2f}")
        p(f"- Converged within the iteration limit: ours {sum(bool(r.get('converged')) for r in full)}/{len(full)}")
        p("")

    by_id = {label: {r["id"]: r for r in (records or [])} for label, records in runs.items()}

    def f1_of(label, entry_id):
        r = by_id[label].get(entry_id)
        return r["metrics"]["f1_score"] if r and r.get("metrics") else None

    p("## Entries discussed in the paper (F1)\n")
    p("| Entry | Paper zero-shot → full | Ours zero-shot → full |")
    p("|---|---|---|")
    for entry_id, (zs, fv) in PAPER_IMPROVEMENTS.items():
        p(f"| {entry_id} | {zs:.3f} → {fv:.3f} | {fmt(f1_of('Zero-shot', entry_id), 3)} → "
          f"{fmt(f1_of('Full (vision)', entry_id), 3)} |")
    p("")
    p("| Entry | Paper full-pipeline F1 | Ours |")
    p("|---|---|---|")
    for entry_id, fv in PAPER_FAILURES.items():
        p(f"| {entry_id} | {fv:.3f} | {fmt(f1_of('Full (vision)', entry_id), 3)} |")
    p("")

    if full:
        worst = sorted((r for r in full if r.get("metrics")), key=lambda r: r["metrics"]["f1_score"])[:5]
        p("## Our five lowest-F1 entries (full pipeline)\n")
        p("| Entry | F1 | CD | IoU | Iterations |")
        p("|---|---|---|---|---|")
        for r in worst:
            m = r["metrics"]
            p(f"| {r['id']} | {m['f1_score']:.3f} | {m['chamfer_distance']:.2f} | {m['volumetric_iou']:.3f} | "
              f"{r.get('num_iterations', '—')} |")
        p("")

    text = "\n".join(out)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8"))
    report = RESULTS_DIR / "paper_comparison.md"
    report.write_text(text, encoding="utf-8")
    print(f"\nSaved to {report}")


if __name__ == "__main__":
    main()
