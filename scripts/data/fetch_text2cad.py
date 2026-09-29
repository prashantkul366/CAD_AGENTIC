"""Build a Text2CAD test subset with executable CadQuery references.

Source: CAD-Coder's release of the official Text2CAD test split
(https://huggingface.co/datasets/gudo7208/CAD-Coder, cad_data_test_cot.json; 8,046 entries,
expert-level (L3) Text2CAD descriptions paired with CadQuery code). The card says Apache-2.0,
but the prompts derive from Text2CAD (CC BY-NC-SA 4.0), so treat the subset as CC BY-NC-SA 4.0.
Not committed: data/external/ is git-ignored (docs/research/DATASETS.md).

Writes data/external/text2cad/test_subset.jsonl: id (DeepCAD uid), prompt, reference_code.
The subset is a fixed random sample (seed) of the entries whose reference executes and
yields a valid solid.

    python scripts/data/fetch_text2cad.py --n 500 --workers 6
"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "external" / "text2cad"
URL = "https://huggingface.co/datasets/gudo7208/CAD-Coder/resolve/main/cad_data_test_cot.json"
PLACEHOLDER = "无可用代码"


def extract(entry: dict):
    msgs = entry.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    code = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    m = re.search(r"description:\s*\n(.*)$", user, re.S)
    prompt = (m.group(1) if m else user).strip()
    if not code or PLACEHOLDER in code:
        return None
    code = re.sub(r"^```(?:python)?\s*\n|```\s*$", "", code.strip(), flags=re.M)
    if not re.search(r"^result\s*=", code, re.M) and re.search(r"^r\s*=", code, re.M):
        code = code.rstrip() + "\nresult = r\n"
    uid = str(entry.get("model_path", "")).replace(".pth", "").split("/")[-1]
    return {"id": uid, "prompt": prompt, "reference_code": code, "source": "CAD-Coder release of Text2CAD test (L3)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--local", default=None, help="use an already downloaded cad_data_test_cot.json")
    ap.add_argument("--ids-file", default=None,
                    help="rebuild exactly this subset (e.g. data/text2cad_subset_ids.txt) instead of sampling")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.local) if args.local else OUT / "cad_data_test_cot.json"
    if not raw_path.exists():
        print("downloading", URL)
        with urllib.request.urlopen(URL, timeout=300) as r:
            raw_path.write_bytes(r.read())
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = [x for x in (extract(e) for e in data) if x]
    print(len(data), "entries,", len(rows), "with code")
    if args.ids_file:
        wanted = [l.strip() for l in open(args.ids_file, encoding="utf-8") if l.strip()]
        by_id = {r["id"]: r for r in rows}
        keep = [by_id[i] for i in wanted if i in by_id]
        with open(OUT / "test_subset.jsonl", "w", encoding="utf-8") as f:
            for row in keep:
                f.write(json.dumps(row) + "
")
        print(f"rebuilt {len(keep)}/{len(wanted)} listed entries -> {OUT / 'test_subset.jsonl'}")
        return
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    from rst.kernel import Kernel
    kernel = Kernel(str(OUT / "verify"), timeout=60)

    def ok(row):
        res = kernel.run(row["reference_code"], [], row["id"], export=False, summaries=False)
        return res.success and res.geometry and res.geometry.get("is_valid") and res.geometry.get("volume", 0) > 0

    keep, checked = [], 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        i = 0
        while len(keep) < args.n and i < len(rows):
            batch = rows[i:i + args.workers * 4]
            i += len(batch)
            for row, good in zip(batch, ex.map(ok, batch)):
                checked += 1
                if good and len(keep) < args.n:
                    keep.append(row)
            print(f"checked {checked}, kept {len(keep)}", flush=True)
    with open(OUT / "test_subset.jsonl", "w", encoding="utf-8") as f:
        for row in keep:
            f.write(json.dumps(row) + "\n")
    (OUT / "SOURCE.txt").write_text(
        f"{URL}\nFixed subset: seed={args.seed}, n={len(keep)} of {checked} checked (reference executes, valid solid).\n"
        "Prompts derive from Text2CAD (CC BY-NC-SA 4.0). Do not redistribute; this folder is git-ignored.\n",
        encoding="utf-8")
    print("valid-reference rate among checked:", round(len(keep) / max(checked, 1), 3))


if __name__ == "__main__":
    main()
