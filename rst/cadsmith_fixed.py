"""CADSmith as a baseline, with its verified bugs fixed and the published pipeline otherwise unchanged.

Fixes (see the audit in the research plan):
  1. A Judge API/JSON failure is retried once and then counts as FAIL, not pass.
  2. In no-vision mode the Judge's system prompt no longer describes rendered views.
  3. If a late refinement breaks the code beyond repair, the last program that
     executed is returned instead of no geometry.
Option `no_leak` removes the retrieval entries that contain near-verbatim
solutions to benchmark items (KB1 pipe-elbow and polygon-prism examples, KB2
sweep pattern), for the leak-free comparison.

autofab keeps token counters and the KBs in module globals, so runs are serialised with a lock.
"""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager

_LOCK = threading.Lock()

LEAKED_EXAMPLES = {"Swept Hollow Pipe Elbow (90-degree bend)", "Regular Polygon Prism (Triangular, Hexagonal, etc.)"}
LEAKED_PATTERNS = {"sweep_degenerate_solid"}


def _no_vision_prompt(system: str) -> str:
    """Drop the paragraphs about the rendered image from the Judge prompt."""
    s = re.sub(r"4\. A rendered image of the generated part.*?(?=\n\nIMPORTANT)", "", system, flags=re.S)
    s = re.sub(r"IMPORTANT: The three rendered views.*?(?=\n\nYour evaluation)", "", s, flags=re.S)
    s = s.replace("- Does the rendered image confirm that the constructed features are actually present and correct?\n", "")
    s = re.sub(r"COUNT them in the rendered views\s+and verify", "COUNT them in the code and verify", s)
    s = s.replace("referencing what you see in the image AND the metrics", "referencing the code AND the metrics")
    return s


@contextmanager
def _patched(no_vision: bool, no_leak: bool, coder_model: str | None, judge_model: str | None):
    from autofab import agents, rag_kb1, rag_kb2
    original_eval = agents.evaluate_geometry
    saved = {"system": agents.VALIDATOR_SYSTEM, "examples": rag_kb1.EXAMPLES, "patterns": rag_kb2.PATTERNS,
             "env": {k: os.environ.get(k) for k in ("CODER_MODEL", "JUDGE_MODEL")}}
    try:
        if no_vision:
            agents.VALIDATOR_SYSTEM = _no_vision_prompt(agents.VALIDATOR_SYSTEM)
        if no_leak:
            rag_kb1.EXAMPLES = [e for e in rag_kb1.EXAMPLES if e.name not in LEAKED_EXAMPLES]
            rag_kb2.PATTERNS = [p for p in rag_kb2.PATTERNS if p.id not in LEAKED_PATTERNS]
        if coder_model:
            os.environ["CODER_MODEL"] = coder_model
        if judge_model:
            os.environ["JUDGE_MODEL"] = judge_model

        def evaluate_with_retry(*a, **kw):
            try:
                return original_eval(*a, **kw)
            except Exception:
                time.sleep(2)
                try:
                    return original_eval(*a, **kw)
                except Exception as e:
                    return {"passed": False, "feedback": f"Judge unavailable ({type(e).__name__}); treated as FAIL."}

        agents.evaluate_geometry = evaluate_with_retry
        yield
    finally:
        agents.VALIDATOR_SYSTEM = saved["system"]
        rag_kb1.EXAMPLES = saved["examples"]
        rag_kb2.PATTERNS = saved["patterns"]
        agents.evaluate_geometry = original_eval
        for k, v in saved["env"].items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_cadsmith_fixed(prompt: str, name: str, output_dir: str, use_vision: bool = True, no_leak: bool = False,
                       coder_model: str | None = None, judge_model: str | None = None,
                       max_refinement_iterations: int = 5, max_error_retries: int = 3) -> dict:
    from autofab import agents
    from autofab.pipeline import Pipeline
    with _LOCK, _patched(not use_vision, no_leak, coder_model, judge_model):
        agents.reset_token_usage()
        t0 = time.time()
        pipe = Pipeline(output_dir=output_dir, max_error_retries=max_error_retries,
                        max_refinement_iterations=max_refinement_iterations, verbose=False, use_vision=use_vision)
        res = pipe.run(prompt, name=name)
        usage = agents.get_token_usage()
    final_code = res.final_code
    if res.final_geometry is None:  # fix 3: fall back to the last program that executed
        for it in reversed(res.iterations):
            if it.execution and it.execution.get("success"):
                final_code = it.code
                break
    rounds = [{"iteration": it.iteration, "type": it.iteration_type, "passed": it.passed,
               "error_retries": len(it.error_retries)} for it in res.iterations]
    return {
        "method": "cadsmith_fixed" + ("" if use_vision else "_novision") + ("_noleak" if no_leak else ""),
        "final_code": final_code, "converged": res.converged, "rounds": rounds,
        "usage": {"calls": usage["calls"], "input_tokens": usage["input_tokens"],
                  "output_tokens": usage["output_tokens"], "by_role": {}},
        "llm_calls_reported": res.total_llm_calls, "seconds": time.time() - t0,
    }
