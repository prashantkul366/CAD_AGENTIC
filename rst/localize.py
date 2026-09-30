"""Blame rules: which statement is responsible for each failing requirement.

Three rules, applied per failing requirement i:

  1. Regression   (persistent / global kinds) i was true and turned false at
                  row t; the statement that produced row t broke it.
  2. Last touch   i was never satisfied (or is terminal): blame the last row
                  whose *measurement* for i changed. For size checks the
                  measurement is the value itself; for feature checks it is the
                  signature of the near miss (the feature that best approximates
                  the requirement), so the blamed step is the one that produced
                  the wrong feature, not a later step that added a different
                  feature nearby ("near-miss provenance").
  3. Missing      the measurement never changed: no operation ever produced
                  the feature. Blame an insertion after the last statement.

Blames on the same statement are merged. Ranking: regressions, then last-touch
by earliest row (early mistakes first), then missing features.

Baseline localisers for the attribution experiment (E3) are included:
`random_localize`, `last_statement_localize` and `llm_localize`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .matrix import Trajectory
from .program import Program

RULES_VERSION = "v2"   # bump whenever a blame rule changes (E3 results are reported per version)

RULE_ORDER = {"regression": 0, "global": 1, "last_touch": 2, "missing": 3, "frame": 4}

# Requirements about one specific feature give sharper evidence than whole-part totals.
LOCAL_TYPES = {"hole_at", "boss_at", "hole_count", "boss_count", "bolt_circle", "coaxial", "planar_face_at",
               "material_at", "fillet_count", "single_solid", "valid"}
# Operations that move the whole part without changing its shape.
RIGID_OPS = {"translate", "rotate", "rotateAboutCenter", "moved", "located", "move", "transformed"}


@dataclass
class Blame:
    stmt: Optional[int]                 # statement to edit (None = unknown)
    row: Optional[int]
    rule: str                           # regression | global | last_touch | missing
    req_ids: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    insert: bool = False                # True: add a new statement after `stmt`

    def weight(self, traj: Trajectory) -> float:
        ids = set(self.req_ids)
        return sum(r.weight for r in traj.reqs if r.id in ids)

    def to_dict(self) -> dict:
        return {"stmt": self.stmt, "row": self.row, "rule": self.rule, "req_ids": self.req_ids,
                "reasons": self.reasons, "insert": self.insert}


def _row_stmt(traj: Trajectory, t: int) -> Optional[int]:
    rows = traj.result.rows
    if 0 <= t < len(rows) and rows[t].stmt is not None:
        return rows[t].stmt
    for k in range(t - 1, -1, -1):
        if rows[k].stmt is not None:
            return rows[k].stmt
    for k in range(t + 1, len(rows)):
        if rows[k].stmt is not None:
            return rows[k].stmt
    return None


def _measure_key(m) -> str:
    return json.dumps(m, sort_keys=True, default=str)


def rigid_rows(traj: Trajectory) -> set:
    """Rows produced by moving/rotating the whole part (same volume and face count as the row before)."""
    rows = traj.result.rows
    out = set()
    for t in range(1, len(rows)):
        a, b = rows[t - 1].summary or {}, rows[t].summary or {}
        if rows[t].op in RIGID_OPS and a and b:
            va, vb = a.get("volume") or 0, b.get("volume") or 0
            if abs(va - vb) <= 1e-6 * max(abs(va), 1.0) and a.get("faces") == b.get("faces"):
                out.add(t)
    return out


def _last_change(traj: Trajectory, i: int, skip=frozenset(), before: Optional[int] = None) -> Optional[int]:
    """Last row whose measurement for requirement i changed (rows in `skip` do not count as changes)."""
    last, prev = None, None
    end = traj.T if before is None else before
    for t in range(end):
        m = traj.measure(t, i)
        key = _measure_key(m)
        if t == 0:
            if m not in (None, [], {}, 0, False):
                last = 0
        elif key != prev and t not in skip:
            last = t
        prev = key
    return last


def _untouched_feature(traj: Trajectory, i: int) -> bool:
    """True when a feature-level check reads the same value on every state since the base body and that
    value says the feature is absent: a void that never appeared (material_at present=False), or a face
    that never appeared anywhere near its required position (planar_face_at)."""
    req = traj.reqs[i]
    if req.type == "material_at":
        return not bool(req.params.get("present", True))
    if req.type == "planar_face_at":
        near = traj.measure(traj.T - 1, i) or []
        off = float(req.params.get("offset", 0.0))
        extent = 0.0
        s = traj.result.final_summary or {}
        bb = s.get("bbox") or {}
        extent = max(bb.get("xlen", 0), bb.get("ylen", 0), bb.get("zlen", 0), 1.0)
        return not any(abs(float(x) - off) <= 0.25 * extent for x in near)
    return False


def blame_requirement(traj: Trajectory, i: int) -> list:
    """Primary blame for failing requirement i, plus an alternative when the evidence is ambiguous."""
    tr = traj.track(i)
    rid = traj.reqs[i].id
    T = traj.T
    final_msg = traj.message(T - 1, i) if T else ""
    rigid = rigid_rows(traj)
    if tr.regressions:
        t = tr.regressions[-1]           # the loss after which it never recovered
        b = Blame(_row_stmt(traj, t), t, "regression", [rid],
                  [f"{rid} held until row {t - 1} and was broken at row {t}: {traj.message(t, i)}"])
        if t not in rigid:
            # Long parts are built non-monotonically: a step may remove a feature that a later step is meant
            # to restore. If the feature was touched again after the loss, that later step is the first suspect.
            later = _last_change(traj, i, skip=rigid)
            if later is not None and later > t:
                return [Blame(_row_stmt(traj, later), later, "last_touch", [rid],
                              [f"{rid} was lost at row {t} and row {later} touched it again without restoring it: "
                               f"{traj.message(later, i)}"]), b]
            return [b]
        # broken by a whole-part move: either the move is wrong, or the part was built in the wrong
        # frame and held only by coincidence before the move -> the last shaping step is an alternative
        b.rule = "frame"
        alt_row = _last_change(traj, i, skip=rigid, before=t)
        alts = []
        if alt_row is not None:
            alts.append(Blame(_row_stmt(traj, alt_row), alt_row, "last_touch", [rid],
                              [f"{rid} held only before a whole-part move at row {t}; "
                               f"the last shaping step was row {alt_row}"]))
        return alts + [b]
    if tr.kind == "global" and tr.global_violations and tr.first_true is None:
        t = tr.global_violations[0]
        return [Blame(_row_stmt(traj, t), t, "global", [rid], [f"{rid} fails from row {t}: {final_msg}"])]
    last = _last_change(traj, i)
    if last == 0 and _untouched_feature(traj, i):
        last = None    # nothing after the base body ever touched this feature: it is missing
    if last is not None:
        return [Blame(_row_stmt(traj, last), last, "last_touch", [rid],
                      [f"{rid} is not satisfied; the last operation that changed what it measures is row "
                       f"{last}: {final_msg}"])]
    by_value = value_provenance(traj, i)
    if by_value is not None:
        return [Blame(by_value, None, "missing", [rid],
                      [f"{rid}: the feature never appeared where required; S{by_value} uses its dimensions "
                       f"and position, so it probably builds it in the wrong place: {final_msg}"])]
    last_stmt = _row_stmt(traj, T - 1) if T else None
    return [Blame(last_stmt, T - 1 if T else None, "missing", [rid],
                  [f"{rid}: no operation ever produced this feature: {final_msg}"], insert=True)]


def _close(a: float, vals: set) -> bool:
    return any(abs(a - v) <= 1e-6 * max(1.0, abs(a)) for v in vals)


def value_provenance(traj: Trajectory, i: int):
    """Statement whose numbers match a missing feature's size AND position (e.g. hole(9) with (40, -10)).

    Used only when the matrix says a feature never appeared: the statement that was meant to make it
    is usually still there, building it on the wrong plane or face. Requiring a position match keeps a
    genuinely deleted feature (whose numbers are gone) classified as missing.
    """
    req = traj.reqs[i]
    p = req.params
    if req.type not in ("hole_at", "boss_at", "bolt_circle"):
        return None
    try:
        prog = Program(traj.result.code)
    except SyntaxError:
        return None
    d = float(p.get("diameter", 0) or 0)
    size_keys = [x for x in (d, d / 2.0) if x > 0]
    if req.type == "bolt_circle":
        pd = float(p.get("pitch_diameter", 0) or 0)
        pos_keys = [x for x in (pd, pd / 2.0) if x > 0]
    else:
        pos_keys = [abs(float(c)) for c in (p.get("center") or []) if abs(float(c)) > 1e-9]
    best, best_score = None, 0
    for idx, vals in enumerate(prog.statement_values()):
        if prog.statements[idx].is_param or prog.statements[idx].is_import:
            continue
        absvals = {abs(v) for v in vals}
        if not any(_close(k, absvals) for k in size_keys):
            continue
        pos_hits = sum(1 for k in pos_keys if _close(k, absvals))
        if pos_keys and pos_hits == 0:
            continue
        score = 1 + pos_hits
        if score >= best_score:        # ties -> the later statement
            best, best_score = idx, score
    return best


def localize(traj: Trajectory) -> list:
    """Ranked, merged blames for all failing requirements.

    Ranking: feature-level requirements before whole-part totals; then regressions, near-miss /
    last-touch, missing features, whole-part-move ambiguities; earlier rows first.
    """
    local_of = {r.id: r.type in LOCAL_TYPES for r in traj.reqs}
    blames = []
    for i in traj.failing():
        for b in blame_requirement(traj, i):
            for other in blames:
                if other.stmt == b.stmt and other.insert == b.insert:
                    other.req_ids += [r for r in b.req_ids if r not in other.req_ids]
                    other.reasons += b.reasons
                    if RULE_ORDER[b.rule] < RULE_ORDER[other.rule]:
                        other.rule, other.row = b.rule, b.row
                    break
            else:
                blames.append(b)

    def key(b):
        tier = 0 if any(local_of.get(r) for r in b.req_ids) else 1
        return (tier, RULE_ORDER[b.rule], b.row if b.row is not None else 1e9, -b.weight(traj))

    blames.sort(key=key)
    return blames


# ---------------------------------------------------------------------------
# baselines for the attribution experiment
# ---------------------------------------------------------------------------

def modifying_statements(traj: Trajectory) -> list[int]:
    """Statements that produced at least one state on the result's lineage."""
    seen = []
    for r in traj.result.rows:
        if r.stmt is not None and r.stmt not in seen:
            seen.append(r.stmt)
    return seen


def random_localize(traj: Trajectory, rng: random.Random) -> Optional[int]:
    cands = modifying_statements(traj)
    return rng.choice(cands) if cands else None


def last_statement_localize(traj: Trajectory) -> Optional[int]:
    cands = modifying_statements(traj)
    return cands[-1] if cands else None


LLM_LOCALIZE_SYSTEM = """You are debugging a CadQuery program that builds a 3D part.
The program executes, but some requirements derived from the design request are not satisfied.
Identify the ONE top-level statement most responsible for the failures.
Answer with JSON only: {"statement": "S<number>", "why": "<one sentence>"}"""


def llm_localize(llm, prompt: str, program: Program, traj: Trajectory) -> Optional[int]:
    """Ask the LLM which statement is wrong, given the same final-state evidence."""
    fails = []
    for i in traj.failing():
        r = traj.reqs[i]
        fails.append(f"- {r.id} ({r.type} {json.dumps(r.params)}): {traj.message(traj.T - 1, i) if traj.T else ''}")
    user = (f"DESIGN REQUEST:\n{prompt}\n\nPROGRAM (top-level statements labelled):\n{program.numbered()}\n\n"
            f"FAILING REQUIREMENTS:\n" + "\n".join(fails) +
            f"\n\nFINAL PART MEASUREMENTS:\n{json.dumps(traj.result.final_summary, indent=1)[:4000]}\n\n"
            "Which statement is most responsible?")
    text = llm.complete(LLM_LOCALIZE_SYSTEM, user, role="llm_localize", temperature=0.0, max_tokens=300)
    m = re.search(r"S(\d+)", text)
    return int(m.group(1)) if m else None
