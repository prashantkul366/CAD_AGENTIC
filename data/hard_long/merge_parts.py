"""Merge the per-batch files into one split file (sorted by id, no duplicates).

    python data/hard_long/merge_parts.py                  # parts/part_*.jsonl    -> hard_long_v0.jsonl
    python data/hard_long/merge_parts.py --version v1     # parts_v1/part_*.jsonl -> hard_long_v1.jsonl
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "v0"
    out = HERE / f"hard_long_{version}.jsonl"
    parts = HERE / ("parts" if version == "v0" else f"parts_{version}")
    entries = {}
    for f in [out] + sorted(parts.glob("part_*.jsonl")):
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                entries.setdefault(e["id"], e)
    with open(out, "w", encoding="utf-8") as f:
        for eid in sorted(entries):
            f.write(json.dumps(entries[eid]) + "\n")
    print(f"{len(entries)} entries -> {out}")


if __name__ == "__main__":
    main()
