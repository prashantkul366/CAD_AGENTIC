"""Oracle requirement suites derived from a reference solid (no LLM, uses the reference).

This is the "oracle specification" condition: requirements come from the
target geometry instead of the prompt. It bounds what RST can do with a
perfect specification and lets the attribution experiment run without an LLM.
It must never be used as the method's own specification in headline results.
"""

from __future__ import annotations

from .geometry import ShapeAnalysis


def autospec(A: ShapeAnalysis, max_features: int = 12) -> list[dict]:
    reqs: list[dict] = [{"id": "valid", "type": "valid"}]
    if A.topology.n_solids == 1:
        reqs.append({"id": "single", "type": "single_solid"})
    b = A.bbox
    reqs.append({"id": "bbox", "type": "bbox_size", "params": {"x": round(b["xlen"], 3), "y": round(b["ylen"], 3),
                                                               "z": round(b["zlen"], 3), "tol": 0.2}})
    for ax in ("x", "y", "z"):
        reqs.append({"id": f"range_{ax}", "type": "bbox_range",
                     "params": {"axis": ax.upper(), "min": round(b[f"{ax}min"], 3), "max": round(b[f"{ax}max"], 3),
                                "tol": 0.2}})
    if A.volume > 0:
        reqs.append({"id": "volume", "type": "volume", "params": {"value": round(A.volume, 3), "rel_tol": 0.01}})
    reqs.append({"id": "genus", "type": "through_hole_count", "params": {"count": A.topology.genus}})
    holes = sorted(A.holes(True), key=lambda h: -h.diameter)[:max_features]
    for k, h in enumerate(holes):
        reqs.append({"id": f"hole{k}", "type": "hole_at",
                     "params": {"center": [round(float(x), 3) for x in h.point], "diameter": round(h.diameter, 3),
                                "axis": [round(float(x), 4) for x in h.axis], "tol": 0.1}})
    by_d: dict[float, int] = {}
    for h in A.holes(True):
        by_d[round(h.diameter, 2)] = by_d.get(round(h.diameter, 2), 0) + 1
    for d, n in sorted(by_d.items())[:6]:
        reqs.append({"id": f"nholes_{d:g}", "type": "hole_count", "params": {"count": n, "diameter": d, "tol": 0.05}})
    bosses = sorted(A.bosses(True), key=lambda c: -c.diameter)[:max_features // 2]
    for k, c in enumerate(bosses):
        reqs.append({"id": f"boss{k}", "type": "boss_at",
                     "params": {"center": [round(float(x), 3) for x in c.point], "diameter": round(c.diameter, 3),
                                "axis": [round(float(x), 4) for x in c.axis], "tol": 0.1}})
    # the extreme planar faces along each axis (top/bottom etc.)
    for normal, key in (("+Z", "zmax"), ("-Z", "zmin"), ("+X", "xmax"), ("-X", "xmin"), ("+Y", "ymax"), ("-Y", "ymin")):
        offset = b[key]
        base = _axis(normal[1:])
        if any(abs(float(f.center @ base) - offset) < 1e-3 and f.normal @ _axis(normal) > 0.999 for f in A.planes):
            reqs.append({"id": f"face{normal}", "type": "planar_face_at",
                         "params": {"normal": normal, "offset": round(offset, 3), "tol": 0.1}})
    for r in reqs:
        r["source"] = "oracle"
    return reqs


def _axis(name: str):
    import numpy as np
    from .geometry import parse_direction
    return np.asarray(parse_direction(name))
