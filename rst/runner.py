"""Subprocess entry point: execute a program with tracing and evaluate requirements.

Usage (called by rst.kernel.Kernel):
    python -m rst.runner job.json

The job file names the script, the requirements, export paths and the output
JSON path. Everything runs in this process so that shapes never need to be
serialised: the parent only receives measurements, verdicts and file paths.
"""

from __future__ import annotations

import json
import sys
import time
import traceback


def _legacy_geometry(A) -> dict:
    """Same fields as the CADSmith executor, computed on the full final shape."""
    t = A.topology
    return {
        "volume": A.volume,
        "center_of_mass": [float(x) for x in A.center_of_mass],
        "bounding_box": A.bbox,
        "is_valid": A.is_valid,
        "num_faces": t.n_faces,
        "num_edges": t.n_edges,
        "num_vertices": t.n_vertices,
        "num_solids": t.n_solids,
        "genus": t.genus,
    }


def run_job(job: dict) -> dict:
    import cadquery as cq
    from .geometry import ShapeAnalysis
    from .program import Program
    from .requirements import evaluate, load_requirements
    from .tracer import FeatureTracer, find_result, final_shape

    script = job["script_path"]
    out = {"success": False}
    t0 = time.time()
    try:
        with open(script, encoding="utf-8") as f:
            src = f.read()
        code = compile(src, script, "exec")
        ns = {"__name__": "__main__"}
        with FeatureTracer(script) as tracer:
            exec(code, ns)
        t_exec = time.time()

        result = find_result(ns)
        if result is None:
            return {"success": False, "error_type": "NoResultError",
                    "error": "No CadQuery Workplane object found. Assign the final shape to a variable named 'result'.",
                    "time_ms": 1000 * (time.time() - t0)}
        events = tracer.timeline(result)
        fshape = events[-1].shape if events and events[-1].wp is result else final_shape(result)
        if fshape is None:
            return {"success": False, "error_type": "NoSolidError",
                    "error": "The result contains no solid (only sketches, wires or faces).",
                    "time_ms": 1000 * (time.time() - t0)}

        try:
            prog = Program(src)
        except SyntaxError:
            prog = None

        rows = []
        for e in events:
            stmt = prog.statement_at_line(e.lineno) if prog else None
            rows.append({"event": e.order, "op": e.op, "lineno": e.lineno, "inner_lineno": e.inner_lineno,
                         "stmt": stmt, "shape": e.shape})
        if not rows or not rows[-1]["shape"].wrapped.IsSame(fshape.wrapped):
            rows.append({"event": None, "op": "final", "lineno": None, "inner_lineno": None,
                         "stmt": rows[-1]["stmt"] if rows else None, "shape": fshape})

        # cap the number of evaluated rows: keep the last event of each statement, then subsample
        max_rows = int(job.get("max_rows", 60))
        n_events = len(rows)
        if len(rows) > max_rows:
            keep = [i for i in range(len(rows)) if i == len(rows) - 1 or rows[i]["stmt"] != rows[i + 1]["stmt"]]
            if len(keep) > max_rows:
                step = len(keep) / max_rows
                keep = sorted({keep[int(k * step)] for k in range(max_rows - 1)} | {keep[-1]})
            rows = [rows[i] for i in keep]

        reqs, errors = load_requirements(job.get("requirements") or [])
        want_summary = job.get("summaries", True)
        out_rows = []
        final_A = None
        for i, r in enumerate(rows):
            A = ShapeAnalysis(r["shape"])
            verdicts = [evaluate(q, A).to_dict() for q in reqs]
            row = {k: r[k] for k in ("event", "op", "lineno", "inner_lineno", "stmt")}
            row["verdicts"] = verdicts
            if want_summary or i == len(rows) - 1:
                row["summary"] = A.summary()
            out_rows.append(row)
            if i == len(rows) - 1:
                final_A = A
        t_eval = time.time()

        if final_A is None:
            final_A = ShapeAnalysis(fshape)
        out = {
            "success": True,
            "rows": out_rows,
            "requirement_ids": [q.id for q in reqs],
            "requirement_errors": errors,
            "geometry": _legacy_geometry(final_A),
            "final_summary": final_A.summary(),
            "coverage": {"n_events": n_events, "n_rows": len(out_rows),
                         "lineage": bool(events), "n_statements": len(prog.statements) if prog else None},
        }
        if job.get("step_path"):
            cq.exporters.export(fshape, job["step_path"])
            out["step_path"] = job["step_path"]
        if job.get("stl_path"):
            cq.exporters.export(fshape, job["stl_path"])
            out["stl_path"] = job["stl_path"]
        out["time_ms"] = 1000 * (time.time() - t0)
        out["exec_ms"] = 1000 * (t_exec - t0)
        out["eval_ms"] = 1000 * (t_eval - t_exec)
    except Exception as e:
        out = {"success": False, "error": traceback.format_exc(), "error_type": type(e).__name__,
               "time_ms": 1000 * (time.time() - t0)}
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    with open(argv[0], encoding="utf-8") as f:
        job = json.load(f)
    out = run_job(job)
    with open(job["out_path"], "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
