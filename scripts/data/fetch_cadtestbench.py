"""Fetch CADTestBench (tests + prompts, MIT) and the CADPrompt reference programs.

Writes data/external/cadtestbench/{detailed,abstract}.jsonl with, per sample:
  id, prompt, reference_code (from CADPrompt Python_Code.py, exporter call
  replaced by `result = <var>`), cadtests (list of test dicts).

Sources:
  https://huggingface.co/datasets/dimitrismallis/CADTestBench        (MIT)
  https://github.com/Kamel773/CAD_Code_Generation (CADPrompt, ICLR 2025; no licence file)
Nothing here is committed: data/external/ is git-ignored. See docs/research/DATASETS.md.

    python scripts/data/fetch_cadtestbench.py
"""

import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "cadtestbench"
HF = "https://huggingface.co/datasets/dimitrismallis/CADTestBench/resolve/main"
RAW = "https://raw.githubusercontent.com/Kamel773/CAD_Code_Generation/main/CADPrompt"


def get(url: str, retries: int = 4) -> bytes:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def reference_to_result(code: str) -> str:
    """Replace `cq.exporters.export(var, ...)` with `result = var`."""
    lines, var = [], None
    for line in code.replace("\r\n", "\n").split("\n"):
        m = re.match(r"\s*(?:cq\.)?exporters\.export\(\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*,", line)
        if m:
            var = var or m.group(1)
            continue
        if re.search(r"\bshow_object\(|ocp_vscode|\bshow\(", line):
            continue
        lines.append(line)
    code = "\n".join(lines).rstrip() + "\n"
    if var and not re.search(r"^result\s*=", code, re.M):
        code += f"result = {var}\n"
    return code


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    tests_all = {}
    for part in ("detailed", "abstract"):
        samples = pd.read_parquet(io.BytesIO(get(f"{HF}/samples/{part}.parquet")))
        tests = pd.read_parquet(io.BytesIO(get(f"{HF}/cadtests/{part}.parquet")))
        tests_all[part] = (samples, tests)
        print(part, len(samples), "samples", len(tests), "tests")
    ids = sorted(set(tests_all["detailed"][0]["sample_id"]) | set(tests_all["abstract"][0]["sample_id"]))
    refs = {}
    for i, sid in enumerate(ids):
        try:
            refs[sid] = reference_to_result(get(f"{RAW}/{sid}/Python_Code.py").decode("utf-8", "replace"))
        except Exception as e:
            print("no reference for", sid, e)
        if i % 25 == 0:
            print(f"references {i}/{len(ids)}")
    for part, (samples, tests) in tests_all.items():
        by_sample = {}
        for rec in tests.to_dict("records"):
            by_sample.setdefault(rec["sample_id"], []).append({k: (v if not hasattr(v, "item") else v.item())
                                                               for k, v in rec.items()})
        with open(OUT / f"{part}.jsonl", "w", encoding="utf-8") as f:
            for rec in samples.to_dict("records"):
                sid = rec["sample_id"]
                f.write(json.dumps({"id": sid, "prompt": rec["prompt"], "partition": part,
                                    "reference_code": refs.get(sid), "cadtests": by_sample.get(sid, [])}) + "\n")
    (OUT / "SOURCE.txt").write_text(
        "CADTestBench tests and prompts: https://huggingface.co/datasets/dimitrismallis/CADTestBench (MIT)\n"
        "Reference programs: CADPrompt, https://github.com/Kamel773/CAD_Code_Generation (no licence file)\n"
        "Do not redistribute; this folder is git-ignored.\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
