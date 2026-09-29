"""Parent-side kernel: run a program in an isolated subprocess and collect its trajectory."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
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
    autospec: Optional[list] = None

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


class _Worker:
    """One long-lived `python -m rst.runner --serve` process."""

    def __init__(self, python: str):
        self.proc = subprocess.Popen([python, "-u", "-m", "rst.runner", "--serve"], cwd=str(REPO_ROOT), env=_clean_env(),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     encoding="utf-8", errors="replace", bufsize=1)
        self.lines = queue.Queue()
        self.jobs = 0
        threading.Thread(target=self._pump, daemon=True).start()
        self.lines.get(timeout=180)  # READY

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put("<closed>")

    def run(self, job_path: str, timeout: float) -> bool:
        self.jobs += 1
        self.proc.stdin.write(job_path + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                return False
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                return False
            if line == "<closed>":
                return False
            if line == f"DONE {job_path}":
                return True

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass


class _Pool:
    """Thread-safe pool of persistent workers (size: RST_KERNEL_WORKERS, default 4)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.idle = []
        self.size = int(os.getenv("RST_KERNEL_WORKERS", "4"))
        self.sem = threading.Semaphore(self.size)

    def run(self, python: str, job_path: str, timeout: float, max_jobs: int = 50) -> bool:
        with self.sem:
            with self.lock:
                w = self.idle.pop() if self.idle else None
            if w is None or w.proc.poll() is not None:
                w = _Worker(python)
            ok = w.run(job_path, timeout)
            if ok and w.jobs < max_jobs:
                with self.lock:
                    self.idle.append(w)
            else:
                w.kill()   # timed out, crashed, or recycled
            return ok


_POOL = _Pool()


class Kernel:
    """Executes CadQuery programs in a separate process with tracing and requirement evaluation.

    persistent=True (default; RST_KERNEL_PERSISTENT=0 disables) reuses long-lived worker
    processes so CadQuery is not re-imported for every call; persistent=False starts a fresh
    process per program (used for held-out evaluation).
    """

    def __init__(self, workdir: str, timeout: int = 60, python: Optional[str] = None, max_rows: int = 60,
                 persistent: Optional[bool] = None):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.python = python or sys.executable
        self.max_rows = max_rows
        self.persistent = (os.getenv("RST_KERNEL_PERSISTENT", "1") != "0") if persistent is None else persistent
        self.calls = 0

    def run(self, code: str, requirements: list[Requirement], name: str,
            export: bool = True, summaries: bool = True, autospec: bool = False) -> KernelResult:
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
            "autospec": autospec,
            "step_path": str(self.workdir / f"{name}.step") if export else None,
            "stl_path": str(self.workdir / f"{name}.stl") if export else None,
        }
        job_path.write_text(json.dumps(job), encoding="utf-8")
        t0 = time.time()
        stderr = ""
        if self.persistent:
            if not _POOL.run(self.python, str(job_path), self.timeout) and not out_path.exists():
                return KernelResult(False, code, requirements,
                                    error=f"Execution timed out after {self.timeout} seconds",
                                    error_type="TimeoutError", time_ms=1000 * (time.time() - t0))
        else:
            try:
                proc = subprocess.run([self.python, "-m", "rst.runner", str(job_path)], cwd=str(self.workdir),
                                      env=_clean_env(), capture_output=True, encoding="utf-8", errors="replace",
                                      timeout=self.timeout)
                stderr = proc.stderr or proc.stdout or ""
            except subprocess.TimeoutExpired:
                return KernelResult(False, code, requirements, error=f"Execution timed out after {self.timeout} seconds",
                                    error_type="TimeoutError", time_ms=1000 * (time.time() - t0))
        if not out_path.exists():
            return KernelResult(False, code, requirements,
                                error=(stderr or "runner produced no output")[-4000:],
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
            time_ms=data.get("time_ms", 0.0), autospec=data.get("autospec"),
        )
