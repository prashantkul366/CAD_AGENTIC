"""Prompts and parsers for every LLM role.

All methods (RST and baselines) share these roles, so differences between
methods come from how the evidence is used, not from prompt wording. No
retrieval knowledge base is used: CADSmith's KB1/KB2 contain near-verbatim
solutions to benchmark items.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .program import Program, parses, strip_fences
from .requirements import Requirement, catalogue_text, load_requirements

# ---------------------------------------------------------------------------
# generator (shared initial program)
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = """You are a CAD engineer. Generate a complete, executable Python script using the CadQuery library to create the requested 3D part.

Rules:
1. import cadquery as cq
2. Assign your final shape to a variable called `result` (type: cq.Workplane).
3. Do NOT import or use ocp_vscode, show(), save_screenshot(), or any visualization.
4. Do NOT call cq.exporters — the system handles export.
5. Output ONLY the Python code. No markdown fences. No explanation text."""

THREADED_STYLE = """
6. Write one feature operation per top-level statement, updating a single main variable, for example:
   result = cq.Workplane("XY").box(80, 60, 4)
   result = result.faces(">Z").workplane().pushPoints(pts).hole(3.4)
   result = result.edges("|Z").fillet(2)
   Build auxiliary bodies in their own statements and combine them with result = result.union(other) or result.cut(other)."""


def generator_system(threaded: bool = True) -> str:
    return GENERATOR_SYSTEM + (THREADED_STYLE if threaded else "")


def generate(llm, prompt: str, threaded: bool = True, requirements: Optional[list] = None) -> str:
    user = prompt
    if requirements:
        user += "\n\nThe part will be checked against these requirements:\n" + _req_lines(requirements)
    return strip_fences(llm.complete(generator_system(threaded), user, role="generate"))


# ---------------------------------------------------------------------------
# requirement writer
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You translate a natural-language CAD request into executable geometric requirements. A CAD kernel will evaluate them exactly on the generated solid and on every intermediate construction state.

Rules:
1. Encode only what the request states or directly implies: explicit dimensions, counts, positions, directions and relations. Never invent values the request does not give.
2. Prefer specific, local checks: one hole_at per positioned hole, bolt_circle for equally spaced holes, planar_face_at for faces at given heights, material_at for bores, slots, pockets and for solid regions, coaxial for concentric features, rotational_symmetry with exact=true for tooth/spoke counts.
3. Always include {"type": "valid"} and, unless the request asks for separate bodies, {"type": "single_solid"}.
4. Use the coordinate conventions stated in the request. If placement is not fixed, avoid absolute positions; if orientation is not fixed, use bbox_size with any_orientation=true.
5. Default length tolerance is 0.1 mm. Include a volume check only if you can compute the volume exactly from the stated dimensions (rel_tol 0.02).
6. For material_at, choose points clearly inside the feature (e.g. on a bore axis at mid-depth), never on a face.
7. Leave "kind" empty unless you need to override the type's default.
8. Give each requirement a short "text" quoting the phrase of the request it checks.

Output JSON only:
{"requirements": [{"id": "R1", "type": "...", "params": {...}, "kind": "", "weight": 1, "text": "..."}]}

Requirement types:
"""


def writer_system() -> str:
    return WRITER_SYSTEM + catalogue_text()


def write_requirements(llm, prompt: str, role: str = "write_requirements") -> tuple[list[Requirement], list[str]]:
    text = llm.complete(writer_system(), f"CAD REQUEST:\n{prompt}", role=role, temperature=0.0)
    data = extract_json(text)
    if data is None:
        return [], [f"writer returned no JSON: {text[:200]}"]
    reqs, errors = load_requirements(data)
    for r in reqs:
        r.source = "self"
    return reqs, errors


# ---------------------------------------------------------------------------
# local repair (RST)
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = """You repair ONE step of a CadQuery program. The program runs, but a CAD kernel evaluated requirements from the design request on every construction state and traced the failures below to the statements marked <<< EDIT.

Rules:
- Change only the marked statements. If a feature is missing entirely, you may instead insert new statements right after the last marked statement.
- Do not touch any other statement: they currently satisfy other requirements.
- Keep variable names so the following statements still work; the final shape must still be assigned to `result`.
- Use the measurements before and after the marked step to reason about what the step did.

Output JSON only:
{"edits": {"S<n>": "<complete new code for that statement>"}, "insert_after": {"S<n>": "<new statements>"}, "why": "<one sentence>"}
Omit "edits" or "insert_after" if unused."""


def repair_local(llm, prompt: str, program: Program, edit_set: list[int], failures: list[str],
                 keep: list[str], before: Optional[dict], after: Optional[dict],
                 allow_insert: bool, extra: str = "") -> tuple[Optional[str], str]:
    """Returns (new_code or None, raw_text)."""
    user = (f"DESIGN REQUEST:\n{prompt}\n\n"
            f"PROGRAM (edit only the statements marked <<< EDIT):\n{program.numbered(set(edit_set))}\n\n"
            f"FAILING REQUIREMENTS:\n" + "\n".join(failures) + "\n\n"
            f"REQUIREMENTS THAT CURRENTLY PASS (do not break them):\n" + ("\n".join(keep) or "(none)") + "\n\n"
            f"PART MEASUREMENTS BEFORE THE MARKED STEP:\n{_brief(before)}\n\n"
            f"PART MEASUREMENTS AFTER THE MARKED STEP:\n{_brief(after)}\n")
    if allow_insert:
        user += "\nThe feature appears to be missing: inserting new statements is allowed.\n"
    if extra:
        user += "\n" + extra
    text = llm.complete(REPAIR_SYSTEM, user, role="repair_local")
    data = extract_json(text)
    if not isinstance(data, dict):
        return None, text
    edits, inserts = {}, {}
    allowed = set(edit_set)
    for key, code in (data.get("edits") or {}).items():
        idx = _stmt_index(key)
        if idx in allowed and isinstance(code, str):
            edits[idx] = strip_fences(code)
    for key, code in (data.get("insert_after") or {}).items():
        idx = _stmt_index(key)
        if idx in allowed and isinstance(code, str):
            inserts[idx] = strip_fences(code)
    if not edits and not inserts:
        return None, text
    new_code = program.replace(edits, inserts)
    if parses(new_code):
        return None, text
    return new_code, text


# ---------------------------------------------------------------------------
# whole-program refine (CADTests-style baseline and ablation)
# ---------------------------------------------------------------------------

REFINE_SYSTEM = """You improve a CadQuery program so the part satisfies the design request. A CAD kernel evaluated the part against requirements derived from the request; some of them fail.

Rules:
- Output the COMPLETE corrected program, assigning the final shape to `result`.
- Do not import visualization libraries and do not call cq.exporters.
- Output ONLY the Python code. No markdown fences. No explanation text."""


def refine_whole(llm, prompt: str, code: str, failures: list[str], passing: list[str],
                 log: Optional[str] = None) -> str:
    user = (f"DESIGN REQUEST:\n{prompt}\n\nCURRENT PROGRAM:\n```python\n{code}\n```\n\n"
            f"FAILING REQUIREMENTS:\n" + "\n".join(failures) + "\n\n"
            f"PASSING REQUIREMENTS:\n" + ("\n".join(passing) or "(none)") + "\n")
    if log:
        user += f"\nCONSTRUCTION LOG (kernel state after each operation):\n{log}\n"
    return strip_fences(llm.complete(REFINE_SYSTEM, user, role="refine_whole"))


# ---------------------------------------------------------------------------
# ReAct-style refine (execution feedback, no requirements)
# ---------------------------------------------------------------------------

REACT_SYSTEM = """You are a CAD engineer checking your own CadQuery program. You get the design request, the program, and exact measurements of the part it produced.
If the part fully satisfies the request, reply with exactly: DONE
Otherwise output the COMPLETE corrected program (final shape in `result`), ONLY the Python code, no fences, no explanation."""


def react_step(llm, prompt: str, code: str, summary: dict) -> Optional[str]:
    user = (f"DESIGN REQUEST:\n{prompt}\n\nPROGRAM:\n```python\n{code}\n```\n\n"
            f"MEASUREMENTS OF THE RESULTING PART:\n{json.dumps(summary, indent=1)[:6000]}\n")
    text = strip_fences(llm.complete(REACT_SYSTEM, user, role="react"))
    if text.strip().upper().startswith("DONE"):
        return None
    return text


# ---------------------------------------------------------------------------
# execution-error fixer (shared inner loop)
# ---------------------------------------------------------------------------

FIX_SYSTEM = """You fix a CadQuery program that failed to execute. Keep the design intent and the overall structure; fix only what the error requires.
Assign the final shape to `result`. Output ONLY the corrected Python code, no fences, no explanation."""


def fix_error(llm, prompt: str, code: str, error: str) -> str:
    user = (f"DESIGN REQUEST:\n{prompt}\n\nPROGRAM:\n```python\n{code}\n```\n\n"
            f"ERROR:\n{(error or '')[-3000:]}\n")
    return strip_fences(llm.complete(FIX_SYSTEM, user, role="fix_error"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_json(text: str):
    t = strip_fences(text)
    for candidate in (t, text):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _stmt_index(key) -> Optional[int]:
    m = re.search(r"(\d+)", str(key))
    return int(m.group(1)) if m else None


def _req_lines(reqs) -> str:
    return "\n".join(f"- {r.id}: {r.text or r.type} ({r.type} {json.dumps(r.params)})" for r in reqs)


def req_line(r: Requirement, verdict: Optional[dict] = None) -> str:
    s = f"- {r.id} [{r.type}] {r.text or ''} params={json.dumps(r.params)}"
    if verdict is not None and not verdict.get("passed") and verdict.get("message"):
        s += f"\n    kernel: {verdict['message']}"
    return s


def _brief(summary: Optional[dict]) -> str:
    if not summary:
        return "(no solid yet)"
    keep = {k: summary.get(k) for k in ("valid", "n_solids", "bbox", "volume", "genus", "n_holes_full", "holes", "bosses")}
    return json.dumps(keep, indent=1)[:3000]


def construction_log(result) -> str:
    """Per-state summary lines used by the CADTests+Log-style baseline."""
    lines = []
    for r in result.rows:
        s = r.summary or {}
        bb = s.get("bbox", {})
        lines.append(f"state {r.idx} (statement S{r.stmt}, .{r.op}): bbox {bb.get('xlen')}x{bb.get('ylen')}x{bb.get('zlen')} "
                     f"volume {s.get('volume')} solids {s.get('n_solids')} holes {s.get('n_holes_full')}")
    return "\n".join(lines)
