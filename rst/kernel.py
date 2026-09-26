"""Parent-side kernel: run a program in an isolated subprocess and collect its trajectory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent

# Environment variables never passed to generated code (credentials for the LLM backends).
_SECRET_PREFIXES = ("AWS_", "ANTHROPIC_", "OPENAI_", "HF_TOKEN", "HUGGING")


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(_SECRET_PREFIXES)}
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


@dataclass
class Row:
    idx: int
    stmt: Optional[int]
    op: str
    lineno: Optional[int]
    verdicts: list
    summary: Optional[dict] = None


@dataclass
class KernelResult:
    success: bool
    code: str
    requirements: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    geometry: Optional[dict] = None
    final_summary: Optional[dict] = None
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    coverage: dict = field(default_factory=dict)
    requirement_errors: list = field(default_factory=list)
    time_ms: float = 0.0

    # --- matrix views ---------------------------------------------------------------

    @property
    def M(self) -> np.ndarray:
        """Satisfaction matrix: rows = construction states, columns = requirements."""
        if not self.rows:
            return np.zeros((0, len(self.requirements)), dtype=bool)
        return np.array([[bool(v["passed"]) for v in r.verdicts] for r in self.rows], dtype=bool)

    @property
    def final_verdicts(self) -> list:
        return self.rows[-1].verdicts if self.rows else []

    def final_pass(self) -> list[bool]:
        return [bool(v["passed"]) for v in self.final_verdicts]

    def score(self) -> float:
        """Weighted fraction of requirements satisfied by the final state (0 if it did not run)."""
        if not self.success or not self.requirements:
            return 0.0
        w = np.array([r.weight for r in self.requirements], dtype=float)
        p = np.array(self.final_pass(), dtype=float)
        return float((w * p).sum() / max(w.sum(), 1e-9))

    def all_pass(self) -> bool:
        return self.success and bool(self.requirements) and all(self.final_pass())

    def to_dict(self, with_rows: bool = True) -> dict:
        d = {
            "success": self.success, "error": self.error, "error_type": self.error_type,
            "geometry": self.geometry, "final_summary": self.final_summary,
            "step_path": self.step_path, "stl_path": self.stl_path, "coverage": self.coverage,
            "requirement_errors": self.requirement_errors, "time_ms": self.time_ms,
            "score": self.score(), "all_pass": self.all_pass(),
            "final_verdicts": self.final_verdicts,
        }
        if with_rows:
            d["rows"] = [{"idx": r.idx, "stmt": r.stmt, "op": r.op, "lineno": r.lineno,
                          "passed": [bool(v["passed"]) for v in r.verdicts],
                          "measures": [v.get("measure") for v in r.verdicts]} for r in self.rows]
        return d


class Kernel:
    """Executes CadQuery programs in a subprocess with tracing and requirement evaluation."""

    def __init__(self, workdir: str, timeout: int = 60, python: Optional[str] = None, max_rows: int = 60):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.python = python or sys.executable
        self.max_rows = max_rows
        self.calls = 0

    def run(self, code: str, requirements: list[Requirement], name: str,
            export: bool = True, summaries: bool = True) -> KernelResult:
        self.calls += 1
        script = self.workdir / f"{name}.py"
        job_path = self.workdir / f"{name}.job.json"
        out_path = self.workdir / f"{name}.out.json"
        script.write_text(code, encoding="utf-8")
        if out_path.exists():
            out_path.unlink()
        job = {
            "script_path": str(script),
            "requirements": [r.to_dict() for r in requirements],
            "out_path": str(out_path),
            "max_rows": self.max_rows,
            "summaries": summaries,
            "step_path": str(self.workdir / f"{name}.step") if export else None,
            "stl_path": str(self.workdir / f"{name}.stl") if export else None,
        }
        job_path.write_text(json.dumps(job), encoding="utf-8")
        t0 = time.time()
        try:
            proc = subprocess.run([self.python, "-m", "rst.runner", str(job_path)], cwd=str(self.workdir),
                                  env=_clean_env(), capture_output=True, encoding="utf-8", errors="replace",
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return KernelResult(False, code, requirements, error=f"Execution timed out after {self.timeout} seconds",
                                error_type="TimeoutError", time_ms=1000 * (time.time() - t0))
        if not out_path.exists():
            return KernelResult(False, code, requirements,
                                error=(proc.stderr or proc.stdout or "runner produced no output")[-4000:],
                                error_type="SubprocessError", time_ms=1000 * (time.time() - t0))
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if not data.get("success"):
            return KernelResult(False, code, requirements, error=data.get("error"), error_type=data.get("error_type"),
                                time_ms=data.get("time_ms", 1000 * (time.time() - t0)))
        rows = [Row(i, r.get("stmt"), r.get("op", "?"), r.get("lineno"), r.get("verdicts", []), r.get("summary"))
                for i, r in enumerate(data["rows"])]
        return KernelResult(
            success=True, code=code, requirements=requirements, rows=rows,
            geometry=data.get("geometry"), final_summary=data.get("final_summary"),
            step_path=data.get("step_path"), stl_path=data.get("stl_path"),
            coverage=data.get("coverage", {}), requirement_errors=data.get("requirement_errors", []),
            time_ms=data.get("time_ms", 0.0),
        )
