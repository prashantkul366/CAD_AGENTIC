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

RULE_ORDER = {"regression": 0, "global": 1, "last_touch": 2, "missing": 3}


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


def blame_requirement(traj: Trajectory, i: int) -> Blame:
    tr = traj.track(i)
    rid = traj.reqs[i].id
    T = traj.T
    final_msg = traj.message(T - 1, i) if T else ""
    if tr.regressions:
        t = tr.regressions[-1]           # the loss after which it never recovered
        return Blame(_row_stmt(traj, t), t, "regression", [rid],
                     [f"{rid} held until row {t - 1} and was broken at row {t}: {traj.message(t, i)}"])
    if tr.kind == "global" and tr.global_violations and tr.first_true is None:
        t = tr.global_violations[0]
        return Blame(_row_stmt(traj, t), t, "global", [rid], [f"{rid} fails from row {t}: {final_msg}"])
    # last-touch: last row whose measurement changed
    last_change = None
    prev = None
    for t in range(T):
        key = _measure_key(traj.measure(t, i))
        if t == 0:
            if traj.measure(t, i) not in (None, [], {}, 0, False):
                last_change = 0
        elif key != prev:
            last_change = t
        prev = key
    if last_change is not None:
        return Blame(_row_stmt(traj, last_change), last_change, "last_touch", [rid],
                     [f"{rid} is not satisfied; the last operation that changed what it measures is row "
                      f"{last_change}: {final_msg}"])
    last_stmt = _row_stmt(traj, T - 1) if T else None
    return Blame(last_stmt, T - 1 if T else None, "missing", [rid],
                 [f"{rid}: no operation ever produced this feature: {final_msg}"], insert=True)


def localize(traj: Trajectory) -> list[Blame]:
    """Ranked, merged blames for all failing requirements."""
    blames: list[Blame] = []
    for i in traj.failing():
        b = blame_requirement(traj, i)
        for other in blames:
            if other.stmt == b.stmt and other.insert == b.insert:
                other.req_ids += b.req_ids
                other.reasons += b.reasons
                if RULE_ORDER[b.rule] < RULE_ORDER[other.rule]:
                    other.rule, other.row = b.rule, b.row
                break
        else:
            blames.append(b)
    blames.sort(key=lambda b: (RULE_ORDER[b.rule], b.row if b.row is not None else 1e9, -b.weight(traj)))
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
