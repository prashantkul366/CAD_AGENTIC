"""Zip a run for sharing: everything except the bulky work folders.

    python scripts/rst/pack_run.py --exp pilot        # -> runs/pilot.zip
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {"work", "eval_work"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    exp = ROOT / "runs" / args.exp
    out = ROOT / "runs" / f"{args.exp}.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in exp.rglob("*"):
            rel = p.relative_to(exp)
            if p.is_file() and not (set(rel.parts) & SKIP):
                z.write(p, Path(args.exp) / rel)
                n += 1
    print(f"{n} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
