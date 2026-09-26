"""Typed requirement predicates evaluated on kernel states.

A requirement is a predicate over a B-Rep with a tolerance, a weight and a
kind that says how it should behave along a construction trajectory:

  global      must hold on every state that contains a solid (e.g. validity)
  persistent  once it becomes true it should stay true (e.g. a hole at a place)
  terminal    only meaningful on the final state (e.g. overall bounding box)

The catalogue below is the single source of truth: it drives evaluation, JSON
validation, and the documentation shown to the LLM requirement writer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .geometry import ShapeAnalysis, parse_direction, point_line_distance

KINDS = ("global", "persistent", "terminal")
DEFAULT_TOL = 0.1          # mm
DEFAULT_REL_TOL = 0.02     # 2 % for volumes / areas


@dataclass
class Requirement:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    kind: str = ""
    weight: float = 1.0
    text: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Requirement":
        spec = CATALOGUE.get(d.get("type", ""))
        kind = d.get("kind") or (spec.default_kind if spec else "terminal")
        return Requirement(
            id=str(d.get("id", "")),
            type=str(d.get("type", "")),
            params=dict(d.get("params") or {}),
            kind=kind if kind in KINDS else (spec.default_kind if spec else "terminal"),
            weight=float(d.get("weight", 1.0) or 1.0),
            text=str(d.get("text", "")),
            source=str(d.get("source", "")),
        )


@dataclass
class Verdict:
    passed: bool
    value: Any = None
    message: str = ""
    measure: Any = None          # what the predicate "reads"; used by the last-touch blame rule

    def to_dict(self) -> dict:
        return {"passed": bool(self.passed), "value": _jsonable(self.value),
                "message": self.message, "measure": _jsonable(self.measure)}


@dataclass
class PredicateSpec:
    name: str
    fn: Callable[[ShapeAnalysis, dict], Verdict]
    default_kind: str
    required: tuple
    optional: tuple
    doc: str
    example: dict


CATALOGUE: dict[str, PredicateSpec] = {}


def predicate(name: str, kind: str, required: tuple, optional: tuple, doc: str, example: dict):
    def deco(fn):
        CATALOGUE[name] = PredicateSpec(name, fn, kind, required, optional, doc, example)
        return fn
    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _tol(p: dict, value: float = 0.0) -> float:
    t = p.get("tol")
    if t is not None:
        return float(t)
    return max(DEFAULT_TOL, 0.005 * abs(float(value)))


def _q(x: float, step: float) -> float:
    """Quantise a measurement so that noise below `step` does not count as a change."""
    step = max(step, 1e-6)
    return round(round(float(x) / step) * step, 6)


def _vec3(v) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 2:
        a = np.array([a[0], a[1], 0.0])
    return a[:3]


def _axis_filter(feat, axis) -> bool:
    if axis is None:
        return True
    d = parse_direction(axis)
    return abs(abs(float(np.dot(feat.axis, d))) - 1.0) < 1e-3


def _diam_ok(feat, diameter, tol) -> bool:
    return diameter is None or abs(feat.diameter - float(diameter)) <= tol


def _fmt(v) -> str:
    return "(" + ", ".join(f"{float(x):g}" for x in np.round(np.asarray(v, dtype=float), 3)) + ")"


def _similar_size(f, diameter) -> bool:
    return diameter is None or 0.5 * float(diameter) <= f.diameter <= 2.0 * float(diameter)


def _feat_brief(f) -> list:
    return [round(f.diameter, 3)] + [round(float(x), 2) for x in f.point]


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------

@predicate("valid", "global", (), (), "The shape is a valid, closed B-Rep solid (OCCT isValid).", {})
def _p_valid(A, p):
    ok = A.is_valid
    return Verdict(ok, ok, "" if ok else "shape is not a valid solid", ok)


@predicate("single_solid", "persistent", (), (),
           "The part is exactly one connected solid (no loose bodies, no parts touching only along an edge or point).", {})
def _p_single(A, p):
    n = A.topology.n_solids
    return Verdict(n == 1, n, "" if n == 1 else f"found {n} separate solids", n)


@predicate("solid_count", "terminal", ("count",), (), "Exactly `count` separate solids.", {"count": 1})
def _p_solid_count(A, p):
    n = A.topology.n_solids
    return Verdict(n == int(p["count"]), n, f"found {n} solids, expected {p['count']}", n)


# ---------------------------------------------------------------------------
# envelope and size
# ---------------------------------------------------------------------------

@predicate("bbox_size", "terminal", (), ("x", "y", "z", "tol", "any_orientation"),
           "Overall bounding-box lengths along X/Y/Z in mm (give only the ones the prompt fixes). "
           "any_orientation=true compares sorted lengths when the prompt does not fix orientation.",
           {"x": 80, "y": 60, "z": 4})
def _p_bbox_size(A, p):
    b = A.bbox
    got = {"x": b["xlen"], "y": b["ylen"], "z": b["zlen"]}
    want = {k: float(p[k]) for k in ("x", "y", "z") if p.get(k) is not None}
    if not want:
        return Verdict(True, got, "no dimension given", None)
    if p.get("any_orientation"):
        g = sorted(got.values())
        w = sorted(want.values())
        # compare the given lengths against the closest subset of measured lengths
        ok = all(any(abs(gv - wv) <= _tol(p, wv) for gv in g) for wv in w)
        bad = [] if ok else [f"lengths {sorted(round(x, 3) for x in g)} vs wanted {w}"]
    else:
        bad = [f"{k}: {got[k]:.3f} vs {v:.3f}" for k, v in want.items() if abs(got[k] - v) > _tol(p, v)]
        ok = not bad
    step = min(_tol(p, v) for v in want.values()) / 2
    return Verdict(ok, {k: round(v, 4) for k, v in got.items()}, "; ".join(bad),
                   [_q(got[k], step) for k in sorted(got)])


@predicate("bbox_range", "terminal", ("axis",), ("min", "max", "tol"),
           "The part spans [min, max] along one axis (e.g. 'from Z=0 to Z=10').",
           {"axis": "Z", "min": 0, "max": 10})
def _p_bbox_range(A, p):
    ax = str(p["axis"]).upper().lstrip("+-")
    lo, hi = A.bbox[f"{ax.lower()}min"], A.bbox[f"{ax.lower()}max"]
    bad = []
    if p.get("min") is not None and abs(lo - float(p["min"])) > _tol(p, p["min"]):
        bad.append(f"{ax}min {lo:.3f} vs {float(p['min']):.3f}")
    if p.get("max") is not None and abs(hi - float(p["max"])) > _tol(p, p["max"]):
        bad.append(f"{ax}max {hi:.3f} vs {float(p['max']):.3f}")
    step = _tol(p) / 2
    return Verdict(not bad, [round(lo, 4), round(hi, 4)], "; ".join(bad), [_q(lo, step), _q(hi, step)])


@predicate("bbox_center", "terminal", (), ("x", "y", "z", "tol"),
           "Centre of the bounding box (e.g. 'centered at the origin in X and Y' -> x=0, y=0).",
           {"x": 0, "y": 0})
def _p_bbox_center(A, p):
    b = A.bbox
    c = {"x": (b["xmin"] + b["xmax"]) / 2, "y": (b["ymin"] + b["ymax"]) / 2, "z": (b["zmin"] + b["zmax"]) / 2}
    bad = [f"{k}: {c[k]:.3f} vs {float(p[k]):.3f}" for k in ("x", "y", "z")
           if p.get(k) is not None and abs(c[k] - float(p[k])) > _tol(p)]
    step = _tol(p) / 2
    return Verdict(not bad, {k: round(v, 4) for k, v in c.items()}, "; ".join(bad),
                   [_q(c[k], step) for k in ("x", "y", "z")])


@predicate("volume", "terminal", (), ("value", "rel_tol", "min", "max"),
           "Solid volume in mm^3: a target value with relative tolerance, or a [min, max] range.",
           {"value": 19200, "rel_tol": 0.02})
def _p_volume(A, p):
    v = A.volume
    if p.get("value") is not None:
        target = float(p["value"])
        rt = float(p.get("rel_tol", DEFAULT_REL_TOL))
        ok = abs(v - target) <= rt * abs(target)
        msg = "" if ok else f"volume {v:.1f} vs {target:.1f} (+/-{100 * rt:.1f}%)"
    else:
        lo, hi = p.get("min"), p.get("max")
        ok = (lo is None or v >= float(lo)) and (hi is None or v <= float(hi))
        msg = "" if ok else f"volume {v:.1f} outside [{lo}, {hi}]"
    return Verdict(ok, round(v, 3), msg, _q(v, max(abs(v) * 1e-4, 1e-3)))


@predicate("surface_area", "terminal", ("value",), ("rel_tol",),
           "Total surface area in mm^2 with relative tolerance.", {"value": 12000, "rel_tol": 0.03})
def _p_area(A, p):
    a, target = A.area, float(p["value"])
    rt = float(p.get("rel_tol", 0.03))
    ok = abs(a - target) <= rt * abs(target)
    return Verdict(ok, round(a, 3), "" if ok else f"area {a:.1f} vs {target:.1f}", _q(a, max(abs(a) * 1e-4, 1e-3)))


# ---------------------------------------------------------------------------
# holes, bosses and patterns
# ---------------------------------------------------------------------------

def _matching(feats, p, tol):
    return [f for f in feats if _diam_ok(f, p.get("diameter"), tol) and _axis_filter(f, p.get("axis"))
            and (p.get("through") is None or f.through == bool(p["through"]))]


@predicate("hole_count", "persistent", ("count",), ("diameter", "axis", "through", "tol"),
           "Number of complete cylindrical holes/bores (full 360 deg), optionally filtered by diameter, "
           "axis direction and through/blind.", {"count": 4, "diameter": 3.4, "axis": "Z", "through": True})
def _p_hole_count(A, p):
    tol = _tol(p, p.get("diameter") or 0)
    m = _matching(A.holes(True), p, tol)
    n = len(m)
    ok = n == int(p["count"])
    partial = [f for f in A.holes(False) if not f.full and _diam_ok(f, p.get("diameter"), tol)]
    others = sorted({round(f.diameter, 2) for f in A.holes(True) if _axis_filter(f, p.get("axis")) and all(f is not x for x in m)})
    msg = "" if ok else f"found {n} matching holes, expected {p['count']}" + (
        f" ({len(partial)} partial arcs of that size, e.g. cut through an edge)" if partial else "") + (
        f"; other hole diameters present: {others}" if others else "")
    # footprint: holes along the axis of similar size (0.5x-2x the target), whatever their exact diameter
    footprint = sorted(_feat_brief(f) for f in A.holes(False)
                       if _axis_filter(f, p.get("axis")) and _similar_size(f, p.get("diameter")))
    return Verdict(ok, n, msg, footprint)


@predicate("hole_at", "persistent", ("center", "diameter"), ("axis", "through", "depth", "tol"),
           "A complete hole of `diameter` whose axis passes through `center` [x, y, z] (the coordinate along "
           "the axis is ignored). Use one per hole when positions are given.",
           {"center": [25, 20, 0], "diameter": 6, "axis": "Z", "through": True})
def _p_hole_at(A, p):
    return _feature_at(A.holes(True), A.holes(False), p, "hole")


@predicate("boss_count", "persistent", ("count",), ("diameter", "axis", "tol"),
           "Number of complete convex cylinders (shafts, bosses, pins) with the given diameter/axis.",
           {"count": 1, "diameter": 20, "axis": "Z"})
def _p_boss_count(A, p):
    tol = _tol(p, p.get("diameter") or 0)
    m = [f for f in A.bosses(True) if _diam_ok(f, p.get("diameter"), tol) and _axis_filter(f, p.get("axis"))]
    n = len(m)
    footprint = sorted(_feat_brief(f) for f in A.bosses(False)
                       if _axis_filter(f, p.get("axis")) and _similar_size(f, p.get("diameter")))
    return Verdict(n == int(p["count"]), n, "" if n == int(p["count"]) else f"found {n} bosses, expected {p['count']}",
                   footprint)


@predicate("boss_at", "persistent", ("center", "diameter"), ("axis", "height", "tol"),
           "A complete convex cylinder of `diameter` whose axis passes through `center`; optional axial `height`.",
           {"center": [0, 0, 0], "diameter": 28, "axis": "Z", "height": 40})
def _p_boss_at(A, p):
    return _feature_at(A.bosses(True), A.bosses(False), p, "boss")


def _feature_at(full_feats, all_feats, p, label):
    d = float(p["diameter"])
    tol = _tol(p, d)
    target = _vec3(p["center"])
    axis = p.get("axis")
    best, best_dist = None, float("inf")
    for f in full_feats:
        if not _axis_filter(f, axis):
            continue
        dist = point_line_distance(target, f.point, f.axis)
        if dist < best_dist:
            best, best_dist = f, dist
    ok = False
    msg = ""
    if best is not None:
        pos_ok = best_dist <= max(tol, 0.1)
        dia_ok = abs(best.diameter - d) <= tol
        thr_ok = p.get("through") is None or best.through == bool(p["through"])
        dep = p.get("depth", p.get("height"))
        dep_ok = dep is None or abs(best.depth - float(dep)) <= _tol(p, dep)
        ok = pos_ok and dia_ok and thr_ok and dep_ok
        if not ok:
            parts = []
            if not pos_ok:
                parts.append(f"nearest {label} axis is {best_dist:.2f} mm away")
            if not dia_ok:
                parts.append(f"diameter {best.diameter:.3f} vs {d:.3f}")
            if not thr_ok:
                parts.append("through" if best.through else "blind")
            if not dep_ok:
                parts.append(f"depth {best.depth:.3f} vs {float(dep):.3f}")
            msg = f"{label} at {_fmt(target)}: " + ", ".join(parts)
    else:
        partial = [f for f in all_feats if point_line_distance(target, f.point, f.axis) <= max(tol, 0.5)]
        msg = f"no complete {label} found" + (" (only a partial arc there)" if partial else "")
    radius = max(3.0 * d, 5.0)
    nearby = sorted(_feat_brief(f) for f in all_feats if point_line_distance(target, f.point, f.axis) <= radius)
    return Verdict(ok, {"distance": None if best is None else round(best_dist, 4),
                        "diameter": None if best is None else round(best.diameter, 4)}, msg, nearby)


@predicate("through_hole_count", "terminal", ("count",), (),
           "Number of through-openings of the part, from its genus (Euler-Poincare). A plate with 4 drilled "
           "through-holes has 4; blind holes do not count.", {"count": 4})
def _p_genus(A, p):
    g = A.topology.genus
    return Verdict(g == int(p["count"]), g, "" if g == int(p["count"]) else f"genus {g}, expected {p['count']}", g)


@predicate("coaxial", "persistent", ("diameters",), ("axis", "tol"),
           "All complete cylinders (holes or bosses) with these diameters share one axis (e.g. bore and hub).",
           {"diameters": [28, 14], "axis": "Z"})
def _p_coaxial(A, p):
    tol = _tol(p)
    feats = [f for f in A.cylinders if f.full]
    groups = []
    for dia in p["diameters"]:
        g = [f for f in feats if abs(f.diameter - float(dia)) <= _tol(p, dia) and _axis_filter(f, p.get("axis"))]
        if not g:
            return Verdict(False, None, f"no complete cylinder of diameter {dia}", [str(dia), "missing"])
        groups.append(g)
    ref = groups[0][0]
    ok = True
    for g in groups[1:]:
        if not any(np.linalg.norm(np.cross(f.axis, ref.axis)) < 1e-3 and
                   point_line_distance(f.point, ref.point, ref.axis) <= max(tol, 0.1) for f in g):
            ok = False
    return Verdict(ok, None, "" if ok else "cylinders are not coaxial",
                   [_feat_brief(g[0]) for g in groups])


@predicate("bolt_circle", "persistent", ("count", "diameter", "pitch_diameter"), ("center", "axis", "tol"),
           "`count` holes of `diameter`, equally spaced on a circle of `pitch_diameter` around `center` "
           "(default origin) with the given axis (default Z).",
           {"count": 6, "diameter": 6.5, "pitch_diameter": 38, "center": [0, 0, 0], "axis": "Z"})
def _p_bolt_circle(A, p):
    d = float(p["diameter"])
    tol = _tol(p, d)
    axis = parse_direction(p.get("axis", "Z"))
    center = _vec3(p.get("center", [0, 0, 0]))
    R = float(p["pitch_diameter"]) / 2.0
    holes = [f for f in A.holes(True) if abs(f.diameter - d) <= tol and abs(abs(np.dot(f.axis, axis)) - 1) < 1e-3]
    on_circle = []
    for f in holes:
        c = f.center_near(center)
        r = point_line_distance(c, center, axis)
        if abs(r - R) <= max(tol, 0.2):
            on_circle.append(c)
    n = len(on_circle)
    footprint = sorted([round(f.diameter, 2)] + [round(float(x), 2) for x in f.center_near(center)]
                       for f in A.holes(False) if abs(abs(np.dot(f.axis, axis)) - 1) < 1e-3 and _similar_size(f, d)
                       and abs(point_line_distance(f.center_near(center), center, axis) - R) <= max(3 * d, 5.0))
    spacing_ok = True
    if n >= 2:
        # angles in the plane perpendicular to the axis
        u = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ang = sorted(math.atan2(np.dot(c - center, v), np.dot(c - center, u)) % (2 * math.pi) for c in on_circle)
        gaps = np.diff(ang + [ang[0] + 2 * math.pi])
        spacing_ok = bool(np.max(np.abs(gaps - 2 * math.pi / n)) < math.radians(1.0))
    ok = n == int(p["count"]) and spacing_ok
    msg = "" if ok else f"{n} holes on the pitch circle, expected {p['count']}" + ("" if spacing_ok else ", unequal spacing")
    return Verdict(ok, n, msg, footprint)


# ---------------------------------------------------------------------------
# faces and material
# ---------------------------------------------------------------------------

@predicate("planar_face_at", "persistent", ("normal", "offset"), ("min_area", "tol"),
           "A planar face with outward normal `normal` (e.g. '+Z') lying on the plane normal.x = offset "
           "(e.g. the top face at Z=10 -> normal '+Z', offset 10).", {"normal": "+Z", "offset": 10})
def _p_planar(A, p):
    n = parse_direction(p["normal"])
    off = float(p["offset"])
    tol = _tol(p, off)
    min_area = float(p.get("min_area", 0.0))
    cands = [f for f in A.planes if np.dot(f.normal, n) > 0.999]
    hits = [f for f in cands if abs(f.offset - off) <= tol and f.area >= min_area]
    near = sorted(round(f.offset, 3) for f in cands)
    ok = bool(hits)
    msg = "" if ok else f"no {p['normal']} face at {off} (faces with that normal at {near[:8]})"
    return Verdict(ok, near[:8], msg, [_q(f.offset, tol / 2) for f in sorted(cands, key=lambda f: f.offset)])


@predicate("face_count", "terminal", ("face_type",), ("count", "min", "max"),
           "Number of faces of a surface type: PLANE, CYLINDER, CONE, SPHERE, TORUS, BSPLINE, or 'any'.",
           {"face_type": "PLANE", "min": 6})
def _p_face_count(A, p):
    t = str(p["face_type"]).upper()
    ft = A.topology.face_types
    n = sum(ft.values()) if t == "ANY" else ft.get(t, 0)
    ok = True
    if p.get("count") is not None:
        ok = n == int(p["count"])
    if p.get("min") is not None:
        ok = ok and n >= int(p["min"])
    if p.get("max") is not None:
        ok = ok and n <= int(p["max"])
    return Verdict(ok, n, "" if ok else f"{n} {t} faces", n)


@predicate("material_at", "persistent", ("point", "present"), (),
           "Material is present (present=true) or absent (present=false, e.g. inside a bore, slot or pocket) "
           "at the point [x, y, z]. Pick points well inside the feature, not on a boundary.",
           {"point": [0, 0, 30], "present": False})
def _p_material(A, p):
    pt = _vec3(p["point"])
    inside = A.contains(pt)
    want = bool(p["present"])
    ok = inside == want
    return Verdict(ok, inside, "" if ok else f"material {'absent' if want else 'present'} at {_fmt(pt)} "
                   f"but expected {'present' if want else 'absent'}", inside)


# ---------------------------------------------------------------------------
# symmetry
# ---------------------------------------------------------------------------

@predicate("reflection_symmetry", "terminal", ("normal",), ("offset", "tol"),
           "The part is mirror-symmetric about the plane with this normal (e.g. 'X' = the YZ plane) at `offset`.",
           {"normal": "X", "offset": 0})
def _p_reflect(A, p):
    ok, dev = A.is_reflection_symmetric(p["normal"], float(p.get("offset", 0.0)), p.get("tol"))
    return Verdict(ok, round(dev, 4), "" if ok else f"not mirror-symmetric (deviation {dev:.3f} mm)", _q(dev, 0.05))


@predicate("rotational_symmetry", "terminal", ("order",), ("axis", "center", "exact", "tol"),
           "The part is invariant under rotation by 360/order degrees about `axis` through `center` "
           "(gears, bolt circles, spokes). exact=true also requires it NOT to be symmetric at 2x order, "
           "which pins down counts such as the number of teeth.", {"order": 16, "axis": "Z", "exact": True})
def _p_rot(A, p):
    n = int(p["order"])
    axis = p.get("axis", "Z")
    center = _vec3(p.get("center", [0, 0, 0]))
    ok, dev = A.is_rotation_symmetric(axis, center, 2 * math.pi / n, p.get("tol"))
    msg = "" if ok else f"not {n}-fold symmetric (deviation {dev:.3f} mm)"
    if ok and p.get("exact", False):
        higher, dev2 = A.is_rotation_symmetric(axis, center, 2 * math.pi / (2 * n), p.get("tol"))
        if higher:
            ok, msg = False, f"also {2 * n}-fold symmetric, so the count is not {n}"
    return Verdict(ok, round(dev, 4), msg, _q(dev, 0.05))


@predicate("fillet_count", "persistent", (), ("radius", "count", "min", "tol"),
           "Number of rounded blend faces (partial cylinders or tori) of the given radius, e.g. filleted edges.",
           {"radius": 2, "min": 4})
def _p_fillet(A, p):
    r = p.get("radius")
    tol = _tol(p, r or 0)
    n = 0
    for f in A.cylinders:
        if not f.full and (r is None or abs(f.radius - float(r)) <= tol):
            n += f.n_faces
    n += A.topology.face_types.get("TORUS", 0) if r is None else _torus_count(A, float(r), tol)
    ok = True
    if p.get("count") is not None:
        ok = n == int(p["count"])
    if p.get("min") is not None:
        ok = ok and n >= int(p["min"])
    return Verdict(ok, n, "" if ok else f"{n} blend faces of that radius", n)


def _torus_count(A, r, tol) -> int:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    n = 0
    for face in A.shape.Faces():
        if face.geomType() == "TORUS":
            try:
                if abs(BRepAdaptor_Surface(face.wrapped).Torus().MinorRadius() - r) <= tol:
                    n += 1
            except Exception:
                pass
    return n


# ---------------------------------------------------------------------------
# validation, evaluation, catalogue text
# ---------------------------------------------------------------------------

def validate(req: Requirement) -> Optional[str]:
    """Return an error string if the requirement is malformed, else None."""
    spec = CATALOGUE.get(req.type)
    if spec is None:
        return f"unknown type {req.type!r}"
    missing = [k for k in spec.required if req.params.get(k) is None]
    if missing:
        return f"{req.type}: missing {missing}"
    if req.type == "bbox_size" and not any(req.params.get(k) is not None for k in ("x", "y", "z")):
        return "bbox_size: give at least one of x, y, z"
    if req.type == "volume" and req.params.get("value") is None and req.params.get("min") is None \
            and req.params.get("max") is None:
        return "volume: give value or min/max"
    try:
        for key in ("axis", "normal"):
            if req.params.get(key) is not None:
                parse_direction(req.params[key])
        for key in ("center", "point"):
            if req.params.get(key) is not None:
                _vec3(req.params[key])
    except Exception as e:
        return f"{req.type}: {e}"
    return None


def evaluate(req: Requirement, analysis: Optional[ShapeAnalysis]) -> Verdict:
    """Evaluate one requirement on one state (None = no solid yet)."""
    if analysis is None:
        return Verdict(False, None, "no solid", None)
    spec = CATALOGUE.get(req.type)
    if spec is None:
        return Verdict(False, None, f"unknown requirement type {req.type}", None)
    try:
        return spec.fn(analysis, req.params)
    except Exception as e:  # a predicate must never crash the run
        return Verdict(False, None, f"evaluation error: {type(e).__name__}: {e}", None)


def load_requirements(data) -> tuple[list[Requirement], list[str]]:
    """Parse a JSON list (or a {'requirements': [...]} object); return (valid, errors)."""
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict):
        data = data.get("requirements", [])
    reqs, errors = [], []
    for i, d in enumerate(data or []):
        if not isinstance(d, dict):
            errors.append(f"item {i}: not an object")
            continue
        r = Requirement.from_dict(d)
        if not r.id:
            r.id = f"R{i + 1}"
        err = validate(r)
        if err:
            errors.append(f"{r.id}: {err}")
        else:
            reqs.append(r)
    seen = set()
    for r in reqs:  # make ids unique
        base, k = r.id, 1
        while r.id in seen:
            k += 1
            r.id = f"{base}_{k}"
        seen.add(r.id)
    return reqs, errors


def catalogue_text() -> str:
    """Human/LLM-readable description of every predicate type."""
    lines = []
    for spec in CATALOGUE.values():
        params = ", ".join(list(spec.required) + [f"{o}?" for o in spec.optional])
        lines.append(f"- {spec.name}({params}) [default kind: {spec.default_kind}]\n    {spec.doc}\n"
                     f"    example params: {json.dumps(spec.example)}")
    return "\n".join(lines)
