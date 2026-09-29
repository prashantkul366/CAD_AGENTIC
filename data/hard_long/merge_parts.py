"""Merge the per-batch files in data/hard_long/parts/ into hard_long_v0.jsonl (sorted by id, no duplicates).

    python data/hard_long/merge_parts.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    entries = {}
    for f in [HERE / "hard_long_v0.jsonl"] + sorted((HERE / "parts").glob("part_*.jsonl")):
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                entries.setdefault(e["id"], e)
    with open(HERE / "hard_long_v0.jsonl", "w", encoding="utf-8") as f:
        for eid in sorted(entries):
            f.write(json.dumps(entries[eid]) + "\n")
    print(f"{len(entries)} entries -> {HERE / 'hard_long_v0.jsonl'}")


if __name__ == "__main__":
    main()
