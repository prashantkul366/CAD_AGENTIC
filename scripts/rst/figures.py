"""Paper figures from the local experiments (no LLM). Writes PNG + PDF to docs/figures/.

  fig_e3_localisation   held-out E3: first guess / top-3 vs random and last-step baselines, both benchmarks
  fig_e3_by_fault       Hard-Long held-out E3 by fault type
  fig_program_length    construction states per reference program: CADSmith-100, CADTestBench, Hard-Long
  fig_teaser_matrix     satisfaction matrix of one held-out Hard-Long fault, with the injected and blamed steps

Colours: reference palette of the dataviz method (series blue for RST, greys for baselines, status green/red
with symbols for pass/fail). Static light-mode figures for the paper.

    python scripts/rst/figures.py
"""

import json
import math
import random
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
RST_BLUE, RST_BLUE_LIGHT = "#2a78d6", "#86b6ef"
BASE_DARK, BASE_LIGHT = "#898781", "#c3c2b7"
SBFL_GREY = "#52514e"         # strongest baseline: darkest neutral

# E3 records re-scored by scripts/rst/e3_rescore.py (every localiser scored with the same repair region)
FIG_RULES = "v3"
E3_RECORDS = ROOT / "runs" / "e3_rescored" / f"{FIG_RULES}_region{FIG_RULES}"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
CHECK, CROSS = "✓", "✗"
SYMBOL_FONT = "DejaVu Sans"   # Segoe UI has no check / cross glyphs

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def records(base, seeds=(1, 2)):
    rows = []
    for s in seeds:
        f = E3_RECORDS / f"{base}_s{s}" / "records.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return rows


def rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def _grid(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _legend_above(ax, ncol):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False, fontsize=7.2, labelcolor=INK2,
              handlelength=1.2, columnspacing=1.0)


def fig_e3_localisation():
    groups = [("Hard-Long\n(~9.5 steps)", records("hardlong_entry")),
              ("CADTestBench refs\n(~2-3 steps)", records("cadtestbench-detailed_oracle_thr"))]
    series = [("RST matrix, first guess", "matrix_hit", RST_BLUE), ("RST matrix, within top 3", "matrix_top3", RST_BLUE_LIGHT),
              ("SBFL (DStar)", "dstar_hit", SBFL_GREY), ("Last step", "last_hit", BASE_DARK),
              ("Random step", "random_hit", BASE_LIGHT)]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    h, gap = 0.15, 0.025
    for gi, (_, rows) in enumerate(groups):
        n = len(rows)
        for si, (label, key, color) in enumerate(series):
            y = gi + (si - (len(series) - 1) / 2) * (h + gap)
            v = rate(rows, key)
            ax.barh(y, v * 100, height=h, color=color, label=label if gi == 0 else None, edgecolor=SURFACE, linewidth=1)
            end = v * 100
            if key == "matrix_hit":
                lo, hi = wilson(round(v * n), n)
                ax.errorbar(v * 100, y, xerr=[[100 * (v - lo)], [100 * (hi - v)]], fmt="none", ecolor=INK2,
                            elinewidth=0.8, capsize=2)
                end = 100 * hi
            ax.text(end + 1.2, y, f"{100 * v:.0f}%", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g}\nn = {len(rows)} faults" for g, rows in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xlabel("injected faults localised to the right step (%)")
    _grid(ax)
    _legend_above(ax, 5)
    ax.set_title("Which step broke the part? (held-out faults, no target geometry)", loc="left", fontsize=9.5,
                 color=INK, pad=22)
    save(fig, "fig_e3_localisation")


def fig_e3_by_fault():
    rows = records("hardlong_entry")
    names = {"feature_delete": "feature deleted", "placement_shift": "feature moved", "count_change": "count changed",
             "param_shift": "size changed", "wrong_workplane": "wrong plane / face"}
    kinds = sorted(names, key=lambda k: -rate([r for r in rows if r["kind"] == k], "matrix_hit"))
    series = [("RST matrix, first guess", "matrix_hit", RST_BLUE), ("SBFL (DStar)", "dstar_hit", SBFL_GREY),
              ("Last step", "last_hit", BASE_DARK), ("Random step", "random_hit", BASE_LIGHT)]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    h, gap = 0.19, 0.025
    for ki, k in enumerate(kinds):
        sub = [r for r in rows if r["kind"] == k]
        for si, (label, key, color) in enumerate(series):
            y = ki + (si - (len(series) - 1) / 2) * (h + gap)
            v = rate(sub, key)
            ax.barh(y, v * 100, height=h, color=color, label=label if ki == 0 else None, edgecolor=SURFACE, linewidth=1)
            ax.text(v * 100 + 1.2, y, f"{100 * v:.0f}%", va="center", fontsize=7.5,
                    color=INK2 if key == "matrix_hit" else MUTED)
        ax.text(99, ki - (len(series) / 2) * (h + gap) + 0.02, f"n = {len(sub)}", fontsize=7, color=MUTED, ha="right", va="top")
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([names[k] for k in kinds])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("localised to the right step (%)")
    _grid(ax)
    _legend_above(ax, 4)
    ax.set_title("Hard-Long held-out faults by type", loc="left", fontsize=9.5, color=INK, pad=22)
    save(fig, "fig_e3_by_fault")


def fig_program_length():
    data = []
    for label, path in (("CADSmith-100", "runs/e1/cadsmith/rows.jsonl"),
                        ("CADTestBench", "runs/e1/cadtestbench-detailed/rows.jsonl")):
        f = ROOT / path
        if f.exists():
            data.append((label, [json.loads(l)["threaded_stmts_modifying"] for l in open(f, encoding="utf-8") if l.strip()]))
    hl = [json.loads(l)["n_ops"] for l in open(ROOT / "data/hard_long/hard_long_v0.jsonl", encoding="utf-8") if l.strip()]
    data.append(("Hard-Long (ours)", hl))
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    rng = random.Random(0)
    for i, (name, vals) in enumerate(data):
        ys = [i + rng.uniform(-0.18, 0.18) for _ in vals]
        xs = [v + rng.uniform(-0.15, 0.15) for v in vals]
        ax.scatter(xs, ys, s=10, color=RST_BLUE if "ours" in name else BASE_DARK, alpha=0.55, linewidths=0)
        mean = sum(vals) / len(vals)
        ax.plot([mean, mean], [i - 0.3, i + 0.3], color=INK, linewidth=1.5)
        ax.text(mean + 0.25, i - 0.3, f"mean {mean:.1f}  (n={len(vals)})", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([n for n, _ in data])
    ax.invert_yaxis()
    ax.set_xlabel("steps that change the solid (one feature per statement)")
    _grid(ax)
    ax.set_title("Existing benchmarks are too short for step-level attribution", loc="left", fontsize=9.5, color=INK)
    save(fig, "fig_program_length")


def fig_teaser_matrix(entry_id=None):
    """Rebuild one held-out Hard-Long fault that RST blamed exactly, and draw its satisfaction matrix."""
    from rst.datasets import load
    from rst.kernel import Kernel
    from rst.localize import localize
    from rst.matrix import Trajectory
    from rst.mutants import make_mutants, normalised
    from rst.requirements import load_requirements
    cands = [r for r in records("hardlong_entry", seeds=(1,)) if r["matrix_strict"] and 1 <= r["n_failing"] <= 4
             and r["kind"] in ("placement_shift", "param_shift", "count_change")]
    cands.sort(key=lambda r: (r["last_hit"], -r["n_failing"]))   # a mid-program fault the last-step guess misses
    if entry_id:
        cands = [r for r in cands if r["id"] == entry_id] or cands
    entries = {e["id"]: e for e in load("hardlong")}
    k = Kernel(str(ROOT / "runs" / "fig_work"), timeout=180)
    chosen = None
    for pick in cands:
        e = entries[pick["id"]]
        code = normalised(e["reference_code"])
        reqs, _ = load_requirements(e["requirements"])
        ref = k.run(code, reqs, "ref", export=False)
        keep = [q for q, ok in zip(reqs, ref.final_pass()) if ok]
        extent = max(ref.geometry["bounding_box"][a] for a in ("xlen", "ylen", "zlen"))
        mut = next((m for m in make_mutants(code, 8, seed=1, extent=extent)
                    if m.stmt == pick["stmt"] and m.detail == pick["detail"]), None)
        if mut is None:
            continue
        traj = Trajectory(k.run(mut.code, keep, "mut", export=False))
        stmts = [r.stmt for r in traj.result.rows]
        if mut.stmt in stmts and traj.T <= 12:
            chosen = (e, mut, traj, stmts)
            break
    if chosen is None:
        print("no suitable teaser example found")
        return None
    e, mut, traj, stmts = chosen
    blame = localize(traj, FIG_RULES)[0]
    M = traj.M
    T, N = M.shape
    failing = traj.failing()
    first_true = {i: next((t for t in range(T) if M[t, i]), T) for i in range(N)}
    passing = sorted([i for i in range(N) if i not in failing], key=lambda i: first_true[i])
    stair, seen = [], set()
    for i in passing:                        # one requirement per establishment step first: the build "staircase"
        if first_true[i] not in seen:
            stair.append(i)
            seen.add(first_true[i])
    stair += [i for i in passing if i not in stair][: max(0, 9 - len(stair))]
    show = sorted(stair, key=lambda i: first_true[i]) + failing
    ops = [traj.result.rows[t].op for t in range(T)]
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(show) + 1.7))
    for yi, i in enumerate(show):
        for t in range(T):
            ok = bool(M[t, i])
            ax.add_patch(Rectangle((t + 0.06, yi + 0.06), 0.88, 0.88, color=GOOD if ok else CRITICAL))
            ax.text(t + 0.5, yi + 0.52, CHECK if ok else CROSS, fontfamily=SYMBOL_FONT, ha="center", va="center",
                    fontsize=7.5, color="white")
    ft = stmts.index(mut.stmt)
    bt = next((t for t in range(T) if stmts[t] == blame.stmt), None)
    ax.add_patch(Rectangle((ft, -0.2), 1, len(show) + 0.4, fill=False, edgecolor=INK, linewidth=1.8))
    label = "fault injected here" + (" = blamed step" if bt == ft else "")
    ax.text(ft + 0.5, -0.28, label, ha="center", va="bottom", fontsize=8, color=INK)
    if bt is not None and bt != ft:
        ax.add_patch(Rectangle((bt, -0.2), 1, len(show) + 0.4, fill=False, edgecolor=INK2, linewidth=1.2, linestyle="--"))
        ax.text(bt + 0.5, -0.28, "blamed step", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(len(show) - len(failing), color=INK2, linewidth=0.8)
    ax.set_xlim(0, T)
    ax.set_ylim(len(show) + 0.1, -0.9)
    ax.set_xticks([t + 0.5 for t in range(T)])
    ax.set_xticklabels([f"S{s}\n.{o}" for s, o in zip(stmts, ops)], fontsize=6.5)
    ax.set_yticks([yi + 0.5 for yi in range(len(show))])
    ax.set_yticklabels([textwrap.shorten(traj.reqs[i].text or traj.reqs[i].type, 46, placeholder=" ...") for i in show],
                       fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    kind = mut.kind.replace("_", " ")
    ax.text(0, 1.0, f"{e['id']} ({e['family'].replace('_', ' ')}), {kind} injected in S{mut.stmt}\n"
            "requirements (rows) checked after every construction step (columns)", transform=ax.transAxes,
            fontsize=8.5, color=INK, va="bottom", ha="left", linespacing=1.4)
    ax.annotate(f"{CHECK} satisfied    {CROSS} not satisfied    rows below the line fail on the final part",
                xy=(0, 0), xycoords="axes fraction", xytext=(0, -30), textcoords="offset points",
                fontfamily=SYMBOL_FONT, fontsize=7.5, color=INK2, va="top", ha="left")
    save(fig, "fig_teaser_matrix")
    return {"id": e["id"], "true_stmt": mut.stmt, "blamed": blame.stmt, "rule": blame.rule, "kind": mut.kind}


if __name__ == "__main__":
    fig_e3_localisation()
    fig_e3_by_fault()
    fig_program_length()
    print(fig_teaser_matrix())
