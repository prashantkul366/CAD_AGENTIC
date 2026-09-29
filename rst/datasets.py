"""Benchmark loaders. Every entry is a dict with at least: id, prompt, reference_code, dataset.

  cadsmith                 CADSmith dataset_v2 (100 prompts, in repo)
  hardlong                 Hard-Long seed split (ours, in repo: data/hard_long/)
  cadtestbench-detailed    CADTestBench / CADPrompt, detailed prompts   (fetched to data/external/)
  cadtestbench-abstract    CADTestBench / CADPrompt, abstract prompts   (fetched to data/external/)
  text2cad                 Text2CAD test subset with CadQuery references (fetched to data/external/)

Third-party data is not committed (licences: docs/research/DATASETS.md); run
scripts/data/fetch_cadtestbench.py and scripts/data/fetch_text2cad.py first.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTERNAL = DATA / "external"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (see rst/datasets.py for how to fetch it)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(name: str, limit: int = 0, ids=None) -> list[dict]:
    if name == "cadsmith":
        rows = []
        for fn in ("t1_primitives.jsonl", "t2_engineering_parts.jsonl", "t3_complex_parts.jsonl"):
            rows += _jsonl(DATA / "dataset_v2" / fn)
    elif name == "hardlong":
        rows = _jsonl(DATA / "hard_long" / "hard_long_v0.jsonl")
    elif name.startswith("cadtestbench-"):
        rows = _jsonl(EXTERNAL / "cadtestbench" / f"{name.split('-', 1)[1]}.jsonl")
    elif name == "text2cad":
        rows = _jsonl(EXTERNAL / "text2cad" / "test_subset.jsonl")
    else:
        raise ValueError(f"unknown dataset {name}")
    for r in rows:
        r.setdefault("dataset", name)
    if ids:
        ids = set(ids)
        rows = [r for r in rows if r["id"] in ids]
    return rows[:limit] if limit else rows
