"""Fetch the generated programs released with CADTestBench (baselines/ in the GitHub repo, MIT).

Writes data/external/cadtestbench/baselines/<method>/<model>/<partition>/<sample_id>.py
where method is CADTests, CADTests_Log or plain (the top-level baseline folder).

    python scripts/data/fetch_cadtestbench_baselines.py --models Claude-4.6-Sonnet
"""

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "cadtestbench" / "baselines"
TREE = "https://api.github.com/repos/dimitrismallis/CADTestBench/git/trees/main?recursive=1"
RAW = "https://raw.githubusercontent.com/dimitrismallis/CADTestBench/main/"


def get(url, retries=4):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Claude-4.6-Sonnet", "GPT-5.2"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tree = json.loads(get(TREE))["tree"]
    jobs = []
    for x in tree:
        p = x["path"]
        if x["type"] != "blob" or not p.startswith("baselines/") or not p.endswith("gpt_generated.py"):
            continue
        parts = p.split("/")
        if parts[1] in ("CADTests", "CADTests_Log"):
            method, model, partition, sid = parts[1], parts[2], parts[3], parts[5]
        else:
            method, model, partition, sid = "plain", parts[1], parts[2], parts[4]
        if model not in args.models:
            continue
        dest = OUT / method / model / partition.lower() / f"{sid}.py"
        jobs.append((RAW + p, dest))
    print(len(jobs), "files")

    def fetch(job):
        url, dest = job
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(url))

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(fetch, jobs))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
