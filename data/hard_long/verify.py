"""Re-verify every Hard-Long entry with the current code (reference passes its suite, >= 8 states,
3 single-statement mutants all caught). Prints a table and exits non-zero if any entry fails.

    python data/hard_long/verify.py            # check only
    python data/hard_long/verify.py --update   # also refresh n_rows / bbox / volume / mutant_kills in the file
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from check_entry import check


def main():
    update = "--update" in sys.argv
    rows = [json.loads(l) for l in open(HERE / "hard_long_v0.jsonl", encoding="utf-8") if l.strip()]
    bad = 0
    print(f"{'id':8s} {'family':26s} {'ok':3s} {'rows':>4s} {'reqs':>4s} kills")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        for e in rows:
            out = check(e, work)
            bad += not out["ok"]
            if update and out["ok"]:
                e.update({k: out[k] for k in ("n_rows", "volume", "bbox", "mutant_kills")})
                e["n_ops"] = out["n_modifying_statements"]
            print(f"{e['id']:8s} {e['family'][:26]:26s} {'yes' if out['ok'] else 'NO ':3s} {out.get('n_rows', 0):>4} "
                  f"{out.get('requirements', 0):>4} {out.get('mutant_kills', '')} {out.get('why', '')}", flush=True)
    print(f"{len(rows) - bad}/{len(rows)} entries pass")
    if update:
        with open(HERE / "hard_long_v0.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e) + "\n")
        print("updated", HERE / "hard_long_v0.jsonl")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
