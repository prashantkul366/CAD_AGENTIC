"""Check that the configured Claude backend and both models are reachable.

Usage:
    python scripts/check_llm.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autofab import llm


def main():
    info = llm.describe()
    print(json.dumps(info, indent=2))
    client = llm.get_client()
    ok = True
    for role, model in (("coder", info["coder_model"]), ("judge", info["judge_model"])):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Reply with the word OK."}],
            )
            print(f"{role:<6} {model}: {response.content[0].text.strip()!r}")
        except Exception as e:
            ok = False
            print(f"{role:<6} {model}: FAILED - {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
