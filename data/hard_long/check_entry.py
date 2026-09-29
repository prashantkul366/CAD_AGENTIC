"""Verify one Hard-Long entry and (optionally) append it to a JSONL file.

Checks: requirements parse; the reference executes to a valid solid; every
requirement passes on the reference; enough traced construction states; and
the suite is discriminative (single-statement mutants fail at least one
requirement).

    python data/hard_long/check_entry.py entry.json --append data/hard_long/parts/part_A.jsonl
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst.kernel import Kernel
from rst.mutants import make_mutants, normalised
from rst.requirements import load_requirements


def check(entry: dict, work: str, n_mutants: int = 3, min_rows: int = 8) -> dict:
    reqs, errors = load_requirements(entry["requirements"])
    if errors:
        return {"ok": False, "why": f"requirement errors: {errors}"}
    k = Kernel(work, timeout=180)
    res = k.run(entry["reference_code"], reqs, entry["id"], export=False)
    if not res.success:
        return {"ok": False, "why": f"reference failed: {res.error_type}: {(res.error or '')[-800:]}"}
    failing = [(r.id, v["message"]) for r, v in zip(reqs, res.final_verdicts) if not v["passed"]]
    if failing:
        return {"ok": False, "why": f"requirements failing on the reference: {failing}"}
    n_rows = len(res.rows)
    n_stmts = len({r.stmt for r in res.rows if r.stmt is not None})
    code = normalised(entry["reference_code"])
    extent = max(res.geometry["bounding_box"][a] for a in ("xlen", "ylen", "zlen"))
    tried = kills = 0
    survivors = []
    for m in make_mutants(code, 12, seed=0, extent=extent):
        if tried >= n_mutants:
            break
        mr = k.run(m.code, reqs, f"{entry['id']}_mut{tried}", export=False)
        if not mr.success:
            continue
        g0, g1 = res.geometry, mr.geometry
        if abs(g0["volume"] - g1["volume"]) < 1e-6 * max(g0["volume"], 1) and all(
                abs(g0["bounding_box"][a] - g1["bounding_box"][a]) < 1e-6 for a in ("xlen", "ylen", "zlen")):
            continue  # geometrically identical mutant, not informative
        tried += 1
        if mr.all_pass():
            survivors.append(m.detail)
        else:
            kills += 1
    ok = n_rows >= min_rows and tried > 0 and kills == tried
    out = {"ok": ok, "n_rows": n_rows, "n_modifying_statements": n_stmts, "requirements": len(reqs),
           "mutant_kills": f"{kills}/{tried}", "survivors": survivors,
           "volume": round(res.geometry["volume"], 3),
           "bbox": [round(res.geometry["bounding_box"][a], 3) for a in ("xlen", "ylen", "zlen")]}
    if not ok:
        out["why"] = (f"only {n_rows} traced states (need {min_rows})" if n_rows < min_rows
                      else f"suite misses mutants: {survivors}" if survivors else "no informative mutants")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--append", default=None)
    args = ap.parse_args()
    entry = json.loads(Path(args.entry).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        out = check(entry, work)
    print(json.dumps(out, indent=1))
    if out["ok"] and args.append:
        entry.update({"n_ops": out["n_modifying_statements"], "n_rows": out["n_rows"], "volume": out["volume"],
                      "bbox": out["bbox"], "mutant_kills": out["mutant_kills"]})
        Path(args.append).parent.mkdir(parents=True, exist_ok=True)
        existing = set()
        if Path(args.append).exists():
            existing = {json.loads(l)["id"] for l in open(args.append, encoding="utf-8") if l.strip()}
        if entry["id"] in existing:
            print("already present, not appended")
        else:
            with open(args.append, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print("appended to", args.append)


if __name__ == "__main__":
    main()
