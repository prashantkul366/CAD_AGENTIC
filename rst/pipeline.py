"""RST and the baselines it is compared against, on one shared engine.

Every method starts from the same initial program P0 (generated once per
prompt and seed) and is charged for every LLM call, including P0 and the
requirement writer, so comparisons are at equal budget:

  zero_shot   P0 as is
  react       execution feedback + kernel measurements, whole-program rewrite
  tests_log   self-written requirements + construction log, whole-program rewrite
              (a re-implementation of the CADTests+Log idea on our requirement DSL)
  best_of_n   N samples, pick the one that passes most self-written requirements
  rst         self-written requirements on every state -> blame -> local repair,
              accepted only if the requirement score improves without regressions

Ablations of RST are configuration switches (localizer, repair scope, acceptance).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import roles
from .kernel import Kernel, KernelResult
from .llm import LLM, Usage
from .localize import (RULES_VERSION, Blame, last_statement_localize, llm_localize, localize, random_localize,
                       region_chains)
from .matrix import Trajectory
from .program import Program


@dataclass
class RSTConfig:
    max_llm_calls: int = 10          # total, including P0 and the requirement writer
    max_error_fixes: int = 3         # per execution
    candidates_per_blame: int = 2
    localizer: str = "matrix"        # matrix | llm | random | last
    repair_scope: str = "local"      # local | whole
    accept: str = "monotone"         # monotone | improve | always
    max_region_deps: int = 2         # extra statements (definitions) editable with the blamed one
    rules: str = RULES_VERSION       # blame-rule version (rst.localize.RULES); v3 also widens the region to tool-body chains
    return_policy: str = "best"      # best | last
    seed: int = 0


@dataclass
class MethodResult:
    method: str
    final_code: str
    final: Optional[dict]
    rounds: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    kernel_calls: int = 0
    seconds: float = 0.0
    requirements: list = field(default_factory=list)
    requirement_errors: list = field(default_factory=list)
    stopped: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _charge(usage: Usage, pre) -> None:
    """Add the shared calls (P0 generation, requirement writer) made before the method started."""
    for role, inp, out in (pre or [("generate", 0, 0)]):
        usage.add(role, int(inp), int(out), 0.0)


class Budget:
    def __init__(self, usage: Usage, limit: int):
        self.usage = usage
        self.limit = limit

    @property
    def left(self) -> int:
        return self.limit - self.usage.calls

    def ok(self, n: int = 1) -> bool:
        return self.left >= n


class Engine:
    """Shared machinery for one prompt."""

    def __init__(self, prompt: str, kernel: Kernel, llm: LLM, name: str):
        self.prompt = prompt
        self.kernel = kernel
        self.llm = llm
        self.name = name
        self._n = 0

    def _tag(self) -> str:
        self._n += 1
        return f"{self.name}_{self._n:03d}"

    def execute(self, code: str, reqs: list, budget: Budget, max_fixes: int, log: list,
                export: bool = True) -> tuple[KernelResult, str]:
        """Run code; on execution errors ask the shared error fixer (charged to the budget)."""
        res = self.kernel.run(code, reqs, self._tag(), export=export)
        fixes = 0
        while not res.success and fixes < max_fixes and budget.ok():
            fixes += 1
            new = roles.fix_error(self.llm, self.prompt, code, f"{res.error_type}: {res.error}")
            log.append({"step": "fix_error", "error_type": res.error_type, "error": (res.error or "")[-500:]})
            code = new
            res = self.kernel.run(code, reqs, self._tag(), export=export)
        return res, code


def _score(res: Optional[KernelResult]) -> float:
    return res.score() if res is not None and res.success else -1.0


def _regressed(old: KernelResult, new: KernelResult) -> list[str]:
    """Requirement ids that pass in `old` but fail in `new` (final states)."""
    if not (old and old.success and new and new.success):
        return []
    return [r.id for r, a, b in zip(old.requirements, old.final_pass(), new.final_pass()) if a and not b]


def _accept(cfg: RSTConfig, cur: KernelResult, new: KernelResult) -> bool:
    if not new.success:
        return False
    if cfg.accept == "always":
        return True
    if _score(new) <= _score(cur):
        return False
    return cfg.accept == "improve" or not _regressed(cur, new)


def _failure_lines(res: KernelResult, ids: Optional[set] = None) -> list[str]:
    out = []
    for r, v in zip(res.requirements, res.final_verdicts):
        if not v["passed"] and (ids is None or r.id in ids):
            out.append(roles.req_line(r, v))
    return out


def _passing_lines(res: KernelResult) -> list[str]:
    return [roles.req_line(r) for r, v in zip(res.requirements, res.final_verdicts) if v["passed"]]


def _finish(method, best_res, best_code, last_res, last_code, cfg_policy, rounds, usage, kernel, t0, reqs, errors,
            stopped) -> MethodResult:
    res, code = (best_res, best_code) if cfg_policy == "best" else (last_res, last_code)
    return MethodResult(method, code, res.to_dict() if res is not None else None, rounds, usage.to_dict(),
                        kernel.calls, time.time() - t0, [r.to_dict() for r in reqs], errors, stopped)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def run_zero_shot(eng: Engine, p0: str, pre=None) -> MethodResult:
    t0, k0 = time.time(), eng.kernel.calls
    usage = Usage()
    _charge(usage, pre)
    res = eng.kernel.run(p0, [], eng._tag())
    out = MethodResult("zero_shot", p0, res.to_dict(), [], usage.to_dict(), eng.kernel.calls - k0, time.time() - t0)
    return out


def run_react(eng: Engine, p0: str, cfg: RSTConfig, pre=None) -> MethodResult:
    t0 = time.time()
    usage = eng.llm.usage = Usage()
    _charge(usage, pre)
    budget = Budget(usage, cfg.max_llm_calls)
    rounds: list = []
    res, code = eng.execute(p0, [], budget, cfg.max_error_fixes, rounds)
    best_res, best_code = res, code
    stopped = "budget"
    while budget.ok() and res.success:
        new = roles.react_step(eng.llm, eng.prompt, code, res.final_summary or {})
        if new is None:
            stopped = "done"
            rounds.append({"step": "react", "decision": "DONE"})
            break
        new_res, new_code = eng.execute(new, [], budget, cfg.max_error_fixes, rounds)
        rounds.append({"step": "react", "decision": "rewrite", "success": new_res.success, "code": new_code})
        if new_res.success:
            res, code = new_res, new_code
            best_res, best_code = res, code   # no self-check available: best == last successful
    return _finish("react", best_res, best_code, res, code, "best", rounds, usage, eng.kernel, t0, [], [], stopped)


def _write(eng: Engine, budget: Budget, reqs: Optional[list]):
    if reqs is not None:
        return reqs, []
    if not budget.ok():
        return [], ["no budget for requirement writer"]
    return roles.write_requirements(eng.llm, eng.prompt)


def run_tests_log(eng: Engine, p0: str, cfg: RSTConfig, reqs: Optional[list] = None,
                  with_log: bool = True, pre=None) -> MethodResult:
    """Whole-program refinement driven by self-written requirements (CADTests+Log style)."""
    t0 = time.time()
    usage = eng.llm.usage = Usage()
    _charge(usage, pre)
    budget = Budget(usage, cfg.max_llm_calls)
    reqs, errors = _write(eng, budget, reqs)
    rounds: list = []
    res, code = eng.execute(p0, reqs, budget, cfg.max_error_fixes, rounds)
    best_res, best_code = res, code
    stopped = "budget"
    while budget.ok():
        if res.success and res.all_pass():
            stopped = "all_pass"
            break
        if not res.success:
            stopped = "exec_failed"
            break
        new = roles.refine_whole(eng.llm, eng.prompt, code, _failure_lines(res), _passing_lines(res),
                                 roles.construction_log(res) if with_log else None)
        new_res, new_code = eng.execute(new, reqs, budget, cfg.max_error_fixes, rounds)
        rounds.append({"step": "refine_whole", "success": new_res.success, "score": _score(new_res),
                       "regressed": _regressed(res, new_res), "code": new_code})
        if new_res.success:
            res, code = new_res, new_code          # CADTests-style: continue from the latest program
            if _score(res) > _score(best_res):
                best_res, best_code = res, code
    name = "tests_log" if with_log else "tests"
    return _finish(name, best_res, best_code, res, code, cfg.return_policy, rounds, usage, eng.kernel, t0,
                   reqs, errors, stopped)


def run_best_of_n(eng: Engine, p0: str, cfg: RSTConfig, reqs: Optional[list] = None, n: Optional[int] = None,
                  threaded: bool = True, pre=None) -> MethodResult:
    t0 = time.time()
    usage = eng.llm.usage = Usage()
    _charge(usage, pre)
    budget = Budget(usage, cfg.max_llm_calls)
    reqs, errors = _write(eng, budget, reqs)
    rounds: list = []
    best_res, best_code = eng.kernel.run(p0, reqs, eng._tag()), p0
    rounds.append({"step": "sample", "score": _score(best_res)})
    n = n or cfg.max_llm_calls
    while budget.ok() and len(rounds) < n:
        code = roles.generate(eng.llm, eng.prompt, threaded=threaded)
        res = eng.kernel.run(code, reqs, eng._tag())
        rounds.append({"step": "sample", "score": _score(res)})
        if _score(res) > _score(best_res):
            best_res, best_code = res, code
    return _finish("best_of_n", best_res, best_code, best_res, best_code, "best", rounds, usage, eng.kernel, t0,
                   reqs, errors, "budget")


def run_rst(eng: Engine, p0: str, cfg: RSTConfig, reqs: Optional[list] = None, pre=None) -> MethodResult:
    """Requirement-Satisfaction Trajectories: blame the step, repair it, keep only improvements."""
    t0 = time.time()
    usage = eng.llm.usage = Usage()
    _charge(usage, pre)
    budget = Budget(usage, cfg.max_llm_calls)
    reqs, errors = _write(eng, budget, reqs)
    rng = random.Random(cfg.seed)
    rounds: list = []
    cur, code = eng.execute(p0, reqs, budget, cfg.max_error_fixes, rounds)
    best_res, best_code = cur, code
    tried: set = set()
    stopped = "budget"
    while budget.ok():
        if not cur.success:
            stopped = "exec_failed"
            break
        if cur.all_pass():
            stopped = "all_pass"
            break
        traj = Trajectory(cur)
        prog = Program(code)
        blames = _choose_blames(cfg, eng, prog, traj, rng, budget)
        blames = [b for b in blames if _blame_key(b) not in tried]
        if not blames:
            stopped = "no_untried_blame"
            break
        blame = blames[0]
        tried.add(_blame_key(blame))
        accepted = False
        for _ in range(cfg.candidates_per_blame):
            if not budget.ok():
                break
            new_code = _propose(cfg, eng, prog, traj, cur, code, blame)
            entry = {"step": "repair", "blame": blame.to_dict(), "scope": cfg.repair_scope}
            if new_code is None:
                entry["result"] = "unparseable"
                rounds.append(entry)
                continue
            new_res, new_code = eng.execute(new_code, reqs, budget, cfg.max_error_fixes, rounds)
            entry.update({"result": "executed" if new_res.success else "exec_failed", "score": _score(new_res),
                          "prev_score": _score(cur), "regressed": _regressed(cur, new_res), "code": new_code})
            if _accept(cfg, cur, new_res):
                entry["accepted"] = True
                rounds.append(entry)
                cur, code = new_res, new_code
                if _score(cur) > _score(best_res):
                    best_res, best_code = cur, code
                tried = set()            # the program changed: all blames are new again
                accepted = True
                break
            entry["accepted"] = False
            rounds.append(entry)
        if not accepted and not budget.ok():
            break
    return _finish("rst", best_res, best_code, cur, code, cfg.return_policy, rounds, usage, eng.kernel, t0,
                   reqs, errors, stopped)


def _blame_key(b: Blame) -> tuple:
    return (b.stmt, b.insert, tuple(sorted(b.req_ids)))


def _choose_blames(cfg: RSTConfig, eng: Engine, prog: Program, traj: Trajectory, rng, budget) -> list[Blame]:
    blames = localize(traj, cfg.rules)
    if cfg.localizer == "matrix" or not blames:
        return blames
    failing_ids = [r.id for i, r in enumerate(traj.reqs) if i in traj.failing()]
    if cfg.localizer == "random":
        stmt = random_localize(traj, rng)
    elif cfg.localizer == "last":
        stmt = last_statement_localize(traj)
    elif cfg.localizer == "llm" and budget.ok(2):
        stmt = llm_localize(eng.llm, eng.prompt, prog, traj)
    else:
        stmt = None
    return [Blame(stmt, None, "last_touch", failing_ids, ["(localised by " + cfg.localizer + ")"])]


def _propose(cfg: RSTConfig, eng: Engine, prog: Program, traj: Trajectory, cur: KernelResult, code: str,
             blame: Blame) -> Optional[str]:
    ids = set(blame.req_ids)
    failures = _failure_lines(cur, ids)
    if cfg.repair_scope == "whole" or blame.stmt is None:
        return roles.refine_whole(eng.llm, eng.prompt, code, failures, _passing_lines(cur), None)
    edit_set = [blame.stmt] + prog.dependencies(blame.stmt, cfg.max_region_deps, chains=region_chains(cfg.rules))
    edit_set = [s for s in edit_set if 0 <= s < len(prog.statements)]
    rows = cur.rows
    t = blame.row if blame.row is not None else len(rows) - 1
    after = rows[t].summary if 0 <= t < len(rows) else None
    before = rows[t - 1].summary if 1 <= t < len(rows) else None
    extra = "Why these statements were blamed:\n" + "\n".join(f"- {r}" for r in blame.reasons)
    new_code, _ = roles.repair_local(eng.llm, eng.prompt, prog, edit_set, failures, _passing_lines(cur),
                                     before, after, blame.insert or blame.rule == "missing", extra)
    return new_code


METHODS = {
    "zero_shot": lambda eng, p0, cfg, reqs=None, pre=None: run_zero_shot(eng, p0, pre),
    "react": lambda eng, p0, cfg, reqs=None, pre=None: run_react(eng, p0, cfg, pre),
    "tests": lambda eng, p0, cfg, reqs=None, pre=None: run_tests_log(eng, p0, cfg, reqs, with_log=False, pre=pre),
    "tests_log": lambda eng, p0, cfg, reqs=None, pre=None: run_tests_log(eng, p0, cfg, reqs, with_log=True, pre=pre),
    "best_of_n": lambda eng, p0, cfg, reqs=None, pre=None: run_best_of_n(eng, p0, cfg, reqs, pre=pre),
    "rst": lambda eng, p0, cfg, reqs=None, pre=None: run_rst(eng, p0, cfg, reqs, pre=pre),
}
