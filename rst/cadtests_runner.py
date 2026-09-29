"""Subprocess entry point: run CADTestBench tests against a generated program.

Mirrors the benchmark's protocol (arXiv 2605.07807, Sec. 5): each test is a
Python snippet over the CadQuery object `final_result`; `check(cond, pass_msg,
fail_msg)` raises on failure. A program that fails to run counts as invalid and
fails every test.

    python -m rst.cadtests_runner job.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback

PREAMBLE = '''
def check(condition, pass_msg="", fail_msg=""):
    if not condition:
        raise AssertionError(fail_msg or "check failed")
'''


def run_job(job: dict) -> dict:
    import cadquery as cq
    import numpy as np
    from .tracer import find_result

    out = {"success": False, "tests": []}
    t0 = time.time()
    try:
        with open(job["script_path"], encoding="utf-8") as f:
            src = f.read()
        ns = {"__name__": "__main__"}
        exec(compile(src, job["script_path"], "exec"), ns)
        # same rule as the CADTestBench evaluator: the object passed to cq.exporters.export(...),
        # otherwise `final_result`, otherwise our usual result lookup
        result = None
        for m in re.finditer(r"exporters\.export\(\s*([A-Za-z_][A-Za-z0-9_]*)", src):
            cand = ns.get(m.group(1))
            if isinstance(cand, (cq.Workplane, cq.Shape)):
                result = cand
        if result is None:
            result = ns.get("final_result")
        if not isinstance(result, (cq.Workplane, cq.Shape)):
            result = find_result(ns)
        if result is None:
            out.update(error="no result object", error_type="NoResultError")
            return out
        if isinstance(result, cq.Shape):
            result = cq.Workplane("XY").add(result)
    except Exception as e:
        out.update(error=traceback.format_exc()[-2000:], error_type=type(e).__name__)
        return out
    out["success"] = True
    for t in job["tests"]:
        env = {"cq": cq, "cadquery": cq, "math": math, "np": np, "numpy": np, "final_result": result, "result": result}
        try:
            exec(PREAMBLE + "\n" + t["code"], env)
            out["tests"].append({"id": t["id"], "passed": True})
        except AssertionError as e:
            out["tests"].append({"id": t["id"], "passed": False, "message": str(e)[:300]})
        except Exception as e:
            out["tests"].append({"id": t["id"], "passed": False, "message": f"{type(e).__name__}: {str(e)[:300]}"})
    out["time_ms"] = 1000 * (time.time() - t0)
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    with open(argv[0], encoding="utf-8") as f:
        job = json.load(f)
    with open(job["out_path"], "w", encoding="utf-8") as f:
        json.dump(run_job(job), f)


if __name__ == "__main__":
    main()
