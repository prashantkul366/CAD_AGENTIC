"""LLM access for RST roles, with per-role usage accounting.

Uses the project's Claude client (Amazon Bedrock by default, or the Claude
API with LLM_BACKEND=anthropic). Every method under comparison goes through
this class so LLM calls and tokens are counted identically.

`ScriptedLLM` replaces the model with a function for tests and dry runs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    by_role: dict = field(default_factory=dict)

    def add(self, role: str, inp: int, out: int, secs: float):
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.seconds += secs
        r = self.by_role.setdefault(role, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        r["calls"] += 1
        r["input_tokens"] += inp
        r["output_tokens"] += out

    def to_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "seconds": round(self.seconds, 2), "by_role": self.by_role}


def default_model() -> str:
    """RST_MODEL if set, else the project's coder model (paper default: Claude Sonnet 4.5)."""
    from autofab.llm import coder_model, resolve_model
    m = os.getenv("RST_MODEL")
    return resolve_model(m) if m else coder_model()


class LLM:
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = 1.0,
                 max_tokens: int = 8192, usage: Optional[Usage] = None):
        from autofab.llm import resolve_model
        self.model = resolve_model(model) if model else default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = usage or Usage()

    def complete(self, system: str, user: str, role: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        from autofab.llm import get_client, response_text
        kwargs = dict(model=self.model, max_tokens=max_tokens or self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        t = self.temperature if temperature is None else temperature
        if t is not None:
            kwargs["temperature"] = t
        t0 = time.time()
        response = get_client().messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        self.usage.add(role, getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0,
                       time.time() - t0)
        return response_text(response)


class ScriptedLLM(LLM):
    """Deterministic stand-in: `fn(role, system, user) -> str`."""

    def __init__(self, fn: Callable[[str, str, str], str], usage: Optional[Usage] = None):
        self.model = "scripted"
        self.temperature = None
        self.max_tokens = 0
        self.usage = usage or Usage()
        self.fn = fn

    def complete(self, system: str, user: str, role: str, temperature=None, max_tokens=None) -> str:
        out = self.fn(role, system, user)
        self.usage.add(role, len(system + user) // 4, len(out) // 4, 0.0)
        return out
