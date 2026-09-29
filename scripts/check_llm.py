"""Check that the configured Claude backend and models are reachable.

Usage:
    python scripts/check_llm.py                             # coder + judge models from .env
    python scripts/check_llm.py --model claude-sonnet-4-6   # also check the model used by RST runs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autofab import llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="extra model id(s) to check")
    args = ap.parse_args()
    info = llm.describe()
    print(json.dumps(info, indent=2))
    client = llm.get_client()
    ok = True
    checks = [("coder", info["coder_model"]), ("judge", info["judge_model"])]
    checks += [(f"extra:{m}", llm.resolve_model(m)) for m in args.model]
    for role, model in checks:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Reply with the word OK."}],
            )
            print(f"{role:<24} {model}: {llm.response_text(response).strip()!r}")
        except Exception as e:
            ok = False
            print(f"{role:<24} {model}: FAILED - {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
