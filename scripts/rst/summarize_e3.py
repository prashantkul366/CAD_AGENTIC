"""Summarise E3 runs (runs/e3/*/records.jsonl) into one markdown table.

Rows: dataset / spec / seed; columns: faults, matrix first blame, matrix top-3, LLM (if run), spectrum-based
fault localisation (Ochiai, DStar; first guess, random tie-breaks), random, last step, with 95 % Wilson intervals for
the matrix rate. A pooled row combines the held-out seeds (>= 1).

    python scripts/rst/summarize_e3.py                    # prints and writes runs/e3/summary.md
    python scripts/rst/summarize_e3.py --dir runs/e3b
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E3 = ROOT / "runs" / "e3"
COLS = ("llm_hit", "ochiai_hit", "dstar_hit", "random_hit", "last_hit")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(rows, key):
    rows = [r for r in rows if key in r]
    return (sum(bool(r[key]) for r in rows), len(rows))


def fmt(k, n):
    return "—" if n == 0 else f"{100 * k / n:.0f}%"


def main():
    global E3
    if "--dir" in sys.argv:
        E3 = ROOT / sys.argv[sys.argv.index("--dir") + 1]
    groups = defaultdict(list)
    for d in sorted(E3.glob("*")):
        f = d / "records.jsonl"
        if not f.exists():
            continue
        tag = d.name
        seed = int(tag.rsplit("_s", 1)[1]) if "_s" in tag else 0
        base = tag.rsplit("_s", 1)[0]
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        rules = (d / "RULES").read_text(encoding="utf-8").strip() if (d / "RULES").exists() else "v1"
        groups[(base, seed, rules)] = rows
    lines = ["| run | rules | seed | faults | matrix first | 95% CI | matrix top-3 | LLM | SBFL Ochiai | SBFL DStar | "
             "random | last step |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pooled = defaultdict(list)
    for (base, seed, rules), rows in sorted(groups.items()):
        k, n = rate(rows, "matrix_hit")
        lo, hi = wilson(k, n)
        cells = [base, rules, str(seed), str(n), fmt(k, n), f"{100 * lo:.0f}–{100 * hi:.0f}%", fmt(*rate(rows, "matrix_top3")),
                 *(fmt(*rate(rows, c)) for c in COLS)]
        lines.append("| " + " | ".join(cells) + " |")
        if seed >= 1:
            pooled[(base.replace("_llm", ""), rules)].extend(rows)
    for (base, rules), rows in sorted(pooled.items()):
        k, n = rate(rows, "matrix_hit")
        lo, hi = wilson(k, n)
        cells = [f"**{base} (held-out pooled)**", rules, "≥1", str(n), f"**{fmt(k, n)}**", f"{100 * lo:.0f}–{100 * hi:.0f}%",
                 fmt(*rate(rows, "matrix_top3")), *(fmt(*rate(rows, c)) for c in COLS)]
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (E3 / "summary.md").write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
