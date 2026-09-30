"""Paired comparisons of localisers on the same E3 faults (the pre-stated tests in docs/research/RST_METHOD.md).

Reads re-scored records (scripts/rst/e3_rescore.py), pools the given seeds, and reports first-guess accuracy with
95 % Wilson intervals, then exact McNemar tests for: matrix (rules B) vs matrix (rules A), and matrix (B) vs each
baseline. Per fault type is descriptive only.

    python scripts/rst/e3_compare.py --base hardlong-v1_entry --seeds 0,1,2
    python scripts/rst/e3_compare.py --base hardlong_entry --seeds 1,2 --a v2_regionv2 --b v3_regionv3
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("dstar_hit", "ochiai_hit", "tarantula_hit", "llm_hit", "random_hit", "last_hit")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(x, y):
    """Exact two-sided McNemar test on paired booleans: (x-only wins, y-only wins, p)."""
    b = sum(1 for a, c in zip(x, y) if a and not c)
    c = sum(1 for a, c2 in zip(x, y) if c2 and not a)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(p, 1.0)


def load(version, base, seeds):
    rows = []
    for s in seeds:
        f = ROOT / "runs" / "e3_rescored" / version / f"{base}_s{s}" / "records.jsonl"
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--a", default="v2_regionv2", help="reference rules (re-scored folder name)")
    ap.add_argument("--b", default="v3_regionv3", help="new rules (re-scored folder name)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    A, B = load(args.a, args.base, seeds), load(args.b, args.base, seeds)
    assert [(r["id"], r["stmt"], r["detail"]) for r in A] == [(r["id"], r["stmt"], r["detail"]) for r in B], \
        "the two versions were not scored on the same faults"
    n = len(B)
    out = [f"## {args.base}, seeds {args.seeds}: {n} faults ({len({r['id'] for r in B})} parts)", "",
           "| localiser | first guess | 95 % CI | top-3 |", "|---|---|---|---|"]

    def line(name, rows, key, top=None):
        k = sum(bool(r[key]) for r in rows)
        lo, hi = wilson(k, len(rows))
        t = f"{100 * sum(bool(r[top]) for r in rows) / len(rows):.1f} %" if top and top in rows[0] else ""
        out.append(f"| {name} | {100 * k / len(rows):.1f} % | {100 * lo:.0f}–{100 * hi:.0f} % | {t} |")

    line(f"matrix ({args.b})", B, "matrix_hit", "matrix_top3")
    line(f"matrix ({args.a})", A, "matrix_hit", "matrix_top3")
    for key in BASELINES:
        if key in B[0]:
            line(key.replace("_hit", "") + f" ({args.b} region)", B, key, key.replace("_hit", "_top3"))
    out += ["", "| paired test (same faults) | only first wins | only second wins | exact McNemar p |", "|---|---|---|---|"]
    tests = [(f"matrix {args.b} vs matrix {args.a}", [r["matrix_hit"] for r in B], [r["matrix_hit"] for r in A])]
    tests += [(f"matrix {args.b} vs {key.replace('_hit', '')}", [r["matrix_hit"] for r in B], [r[key] for r in B])
              for key in BASELINES if key in B[0]]
    for name, x, y in tests:
        b, c, p = mcnemar(x, y)
        out.append(f"| {name} | {b} | {c} | {p:.2g} |")
    out += ["", "| fault type | n | matrix " + args.b + " | matrix " + args.a + " | DStar | Ochiai | random | last |",
            "|---|---|---|---|---|---|---|---|"]
    for kind in sorted({r["kind"] for r in B}):
        sb = [r for r in B if r["kind"] == kind]
        sa = [r for r in A if r["kind"] == kind]
        pct = lambda rows, key: f"{100 * sum(bool(r[key]) for r in rows) / len(rows):.0f} %"
        out.append(f"| {kind} | {len(sb)} | {pct(sb, 'matrix_hit')} | {pct(sa, 'matrix_hit')} | {pct(sb, 'dstar_hit')} | "
                   f"{pct(sb, 'ochiai_hit')} | {pct(sb, 'random_hit')} | {pct(sb, 'last_hit')} |")
    text = "\n".join(out) + "\n"
    dest = ROOT / "runs" / "e3_rescored" / f"compare_{args.base}_s{args.seeds.replace(',', '')}.md"
    dest.write_text(text, encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
