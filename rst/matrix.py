"""Satisfaction-matrix analysis: establishment, regressions, potential and step rewards.

For requirement i and states S_1..S_T (rows of M):
  establishment e_i  first row after which the requirement stays true
  regression         a row where a requirement that was true becomes false
                     (persistent/global kinds only; terminal requirements may
                     legitimately flip before the end)
  potential          Psi(S_t) = sum_i w_i M[t, i]
  step reward        r_t = sum_i w_i [(M_t - M_{t-1})_+ - lambda (M_{t-1} - M_t)_+]
                     with lambda = 1 this telescopes to Psi(S_T) - Psi(S_0)
                     (potential-based shaping, Ng et al. 1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kernel import KernelResult


@dataclass
class RequirementTrack:
    index: int
    req_id: str
    kind: str
    weight: float
    final: bool
    first_true: Optional[int]
    established: Optional[int]
    regressions: list          # rows where it turned false after being true
    global_violations: list    # rows where a global requirement is false

    @property
    def status(self) -> str:
        if self.final:
            return "pass"
        if self.first_true is not None and self.kind in ("persistent", "global"):
            return "regressed"
        return "never" if self.first_true is None else "lost"


class Trajectory:
    def __init__(self, result: KernelResult):
        self.result = result
        self.reqs = result.requirements
        self.M = result.M
        self.T, self.N = self.M.shape
        self.w = np.array([r.weight for r in self.reqs], dtype=float) if self.reqs else np.zeros(0)

    def measure(self, t: int, i: int):
        return self.result.rows[t].verdicts[i].get("measure")

    def message(self, t: int, i: int) -> str:
        return self.result.rows[t].verdicts[i].get("message", "")

    def track(self, i: int) -> RequirementTrack:
        col = self.M[:, i] if self.T else np.zeros(0, dtype=bool)
        req = self.reqs[i]
        first_true = int(np.argmax(col)) if col.any() else None
        established = None
        for t in range(self.T - 1, -1, -1):
            if not col[t]:
                break
            established = t
        regressions = []
        if req.kind in ("persistent", "global") and first_true is not None:
            for t in range(first_true + 1, self.T):
                if col[t - 1] and not col[t]:
                    regressions.append(t)
        global_violations = [t for t in range(self.T) if not col[t]] if req.kind == "global" else []
        return RequirementTrack(i, req.id, req.kind, req.weight, bool(col[-1]) if self.T else False,
                                first_true, established, regressions, global_violations)

    def tracks(self) -> list[RequirementTrack]:
        return [self.track(i) for i in range(self.N)]

    def potential(self) -> np.ndarray:
        if not self.T:
            return np.zeros(0)
        return (self.M.astype(float) * self.w).sum(axis=1)

    def step_rewards(self, lam: float = 1.0) -> np.ndarray:
        """Per-row reward; row 0 is compared against the empty state (all requirements false)."""
        if not self.T:
            return np.zeros(0)
        prev = np.vstack([np.zeros((1, self.N), dtype=bool), self.M[:-1]])
        gain = (self.M & ~prev).astype(float)
        loss = (prev & ~self.M).astype(float)
        return ((gain - lam * loss) * self.w).sum(axis=1)

    def regression_count(self) -> int:
        return sum(len(tr.regressions) for tr in self.tracks())

    def failing(self) -> list[int]:
        return [i for i in range(self.N) if self.T == 0 or not self.M[-1, i]]

    def heatmap_text(self) -> str:
        """Compact text heat map: one line per requirement, one char per state."""
        lines = []
        for i, req in enumerate(self.reqs):
            cells = "".join("#" if self.M[t, i] else "." for t in range(self.T))
            lines.append(f"{req.id:>14s} [{req.kind[0]}] {cells}")
        stmts = "".join(str(r.stmt % 10) if r.stmt is not None else "?" for r in self.result.rows)
        lines.append(f"{'stmt':>14s}     {stmts}")
        return "\n".join(lines)
