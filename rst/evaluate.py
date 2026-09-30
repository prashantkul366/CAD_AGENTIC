"""Held-out evaluation of generated programs.

Three independent evaluators, none of which the methods ever see:

  cadtests   CADTestBench's own tests (Pass Rate, Requirement Score, Invalid Ratio)
  dsl        a held-out requirement suite in our DSL (Hard-Long ships one)
  shape      reference-based shape metrics, both the CADSmith protocol
             (bbox-centre + ICP with scale, shell voxels) and the corrected one
             (rigid ICP, exact OCCT boolean IoU, dimensional errors)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .kernel import Kernel, _clean_env


# ---------------------------------------------------------------------------
# CADTestBench tests
# ---------------------------------------------------------------------------

def run_cadtests(code: Optional[str], tests: list[dict], workdir: str, name: str, timeout: int = 180) -> dict:
    """Pass/fail per test plus the per-sample PR / RS / invalid flags."""
    groups = {}
    for t in tests:
        groups.setdefault(t.get("requirement_id") or t.get("cadtest_id"), []).append(str(t.get("cadtest_id")))
    if not code:
        return _cadtests_summary(False, {}, groups, "no code")
    wd = Path(workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    script, job_path, out_path = wd / f"{name}.py", wd / f"{name}.ctjob.json", wd / f"{name}.ctout.json"
    script.write_text(code, encoding="utf-8")
    if out_path.exists():
        out_path.unlink()
    job = {"script_path": str(script), "out_path": str(out_path),
           "tests": [{"id": str(t.get("cadtest_id")), "code": t["cadtest_code"]} for t in tests]}
    job_path.write_text(json.dumps(job), encoding="utf-8")
    try:
        subprocess.run([sys.executable, "-m", "rst.cadtests_runner", str(job_path)], cwd=str(wd), env=_clean_env(),
                       capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _cadtests_summary(False, {}, groups, "timeout")
    if not out_path.exists():
        return _cadtests_summary(False, {}, groups, "runner produced no output")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    passed = {t["id"]: t["passed"] for t in data.get("tests", [])}
    return _cadtests_summary(data.get("success", False), passed, groups, data.get("error_type"))


def _cadtests_summary(valid: bool, passed: dict, groups: dict, error) -> dict:
    n_groups = len(groups)
    sat = sum(1 for ids in groups.values() if valid and all(passed.get(i, False) for i in ids))
    all_pass = valid and bool(passed) and all(passed.values()) and len(passed) == sum(len(v) for v in groups.values())
    return {"invalid": not valid, "pass": bool(all_pass), "rs": sat / n_groups if n_groups else 0.0,
            "tests_passed": sum(passed.values()) if valid else 0, "tests_total": sum(len(v) for v in groups.values()),
            "error": error}


# ---------------------------------------------------------------------------
# held-out DSL suite
# ---------------------------------------------------------------------------

def run_dsl_suite(kernel: Kernel, code: Optional[str], suite: list, name: str) -> dict:
    from .requirements import load_requirements
    reqs, _ = load_requirements(suite)
    if not code:
        return {"invalid": True, "pass": False, "rs": 0.0}
    res = kernel.run(code, reqs, name, export=False, summaries=False)
    if not res.success:
        return {"invalid": True, "pass": False, "rs": 0.0, "error": res.error_type}
    return {"invalid": False, "pass": res.all_pass(), "rs": res.score(),
            "failed": [r.id for r, ok in zip(reqs, res.final_pass()) if not ok]}


# ---------------------------------------------------------------------------
# shape metrics
# ---------------------------------------------------------------------------

def shape_metrics(gen_stl: str, ref_stl: str, gen_step: Optional[str] = None, ref_step: Optional[str] = None,
                  seeds=(0, 1, 2)) -> dict:
    """CADSmith protocol and corrected protocol on the same pair."""
    from autofab.metrics import compare_stl
    out = {}
    try:
        m = compare_stl(gen_stl, ref_stl, normalize=False)
        out["orig"] = {"cd": m.chamfer_distance, "f1": m.f1_score, "iou": m.volumetric_iou}
    except Exception as e:
        out["orig"] = {"error": str(e)}
    try:
        out["corrected"] = corrected_metrics(gen_stl, ref_stl, gen_step, ref_step, seeds)
    except Exception as e:
        out["corrected"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def corrected_metrics(gen_stl, ref_stl, gen_step=None, ref_step=None, seeds=(0, 1, 2), n_points=10000, tau=1.0) -> dict:
    import trimesh
    from scipy.spatial import cKDTree
    from trimesh.registration import icp

    gen, ref = trimesh.load(gen_stl), trimesh.load(ref_stl)
    cg = (gen.bounds[0] + gen.bounds[1]) / 2
    cr = (ref.bounds[0] + ref.bounds[1]) / 2
    cds, f1s, mats = [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        pg = trimesh.sample.sample_surface(gen, n_points, seed=rng)[0] - cg
        pr = trimesh.sample.sample_surface(ref, n_points, seed=rng)[0] - cr
        # rigid only: no uniform scale, no reflection
        M, _, _ = icp(pg[:5000], pr[:5000], max_iterations=100, scale=False, reflection=False)
        pa = trimesh.transform_points(pg, M)
        d1, _ = cKDTree(pr).query(pa)
        d2, _ = cKDTree(pa).query(pr)
        cds.append(float(np.mean(d1 ** 2) + np.mean(d2 ** 2)))
        p, r = float(np.mean(d1 < tau)), float(np.mean(d2 < tau))
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        mats.append(M)
    # sampling noise floor: reference against itself
    rng = np.random.default_rng(99)
    a = trimesh.sample.sample_surface(ref, n_points, seed=rng)[0]
    b = trimesh.sample.sample_surface(ref, n_points, seed=rng)[0]
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    floor = float(np.mean(da ** 2) + np.mean(db ** 2))
    ext_g, ext_r = np.sort(gen.extents), np.sort(ref.extents)
    out = {
        "cd": float(np.median(cds)), "cd_seeds": cds, "cd_noise_floor": floor, "cd_minus_floor": float(np.median(cds)) - floor,
        "f1": float(np.median(f1s)),
        "bbox_err_mm": float(np.max(np.abs(ext_g - ext_r))),
        "volume_err_pct": float(abs(abs(gen.volume) - abs(ref.volume)) / max(abs(ref.volume), 1e-9) * 100),
    }
    if gen_step and ref_step:
        out.update(exact_iou(gen_step, ref_step, mats[0], cg, cr))
    return out


def text2cad_cd(gen_stl: str, ref_stl: str, n_points: int = 8192, seed: int = 0) -> float:
    """Chamfer distance x 1000 exactly as in Text2CAD's evaluation (Cad_VLM/test.py, CadSeqProc/utility/utils.py):
    each cloud divided by (max - min) over all its coordinates, no centring, no alignment,
    sum of the two mean squared nearest-neighbour distances. Invalid outputs are excluded upstream."""
    import trimesh
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    g = trimesh.sample.sample_surface(trimesh.load(gen_stl), n_points, seed=rng)[0]
    r = trimesh.sample.sample_surface(trimesh.load(ref_stl), n_points, seed=rng)[0]
    g = g / abs(g.max() - g.min())
    r = r / abs(r.max() - r.min())
    d1, _ = cKDTree(g).query(r)
    d2, _ = cKDTree(r).query(g)
    return float((np.mean(d1 ** 2) + np.mean(d2 ** 2)) * 1000.0)


def exact_iou(gen_step: str, ref_step: str, M: np.ndarray, cg, cr) -> dict:
    """IoU = V(A n B) / V(A u B) with OCCT booleans: as placed, and after the rigid alignment."""
    import cadquery as cq
    A = cq.importers.importStep(gen_step).val()
    B = cq.importers.importStep(ref_step).val()
    out = {}
    try:
        out["iou_as_placed"] = _iou(A, B)
    except Exception as e:
        out["iou_as_placed"] = None
    try:
        T = np.eye(4)
        T[:3, 3] = -np.asarray(cg)
        U = np.eye(4)
        U[:3, 3] = np.asarray(cr)
        full = U @ M @ T
        out["iou_aligned"] = _iou(_rigid_move(A, full), B)
    except Exception:
        out["iou_aligned"] = None
    return out


def _rigid_move(shape, full: np.ndarray):
    """Apply a 4x4 rigid transform to an OCCT shape (rotation snapped to the nearest exact rotation)."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    u, _, vt = np.linalg.svd(full[:3, :3])
    R = u @ vt
    if np.linalg.det(R) < 0:          # never introduce a reflection
        u[:, -1] *= -1
        R = u @ vt
    t = full[:3, 3]
    trsf = gp_Trsf()
    trsf.SetValues(*(float(x) for x in (R[0, 0], R[0, 1], R[0, 2], t[0],
                                        R[1, 0], R[1, 1], R[1, 2], t[1],
                                        R[2, 0], R[2, 1], R[2, 2], t[2])))
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape())


def _iou(A, B) -> float:
    inter = A.intersect(B).Volume()
    union = A.Volume() + B.Volume() - inter
    return float(inter / union) if union > 0 else 0.0
