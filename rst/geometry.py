"""Exact kernel measurements of a CadQuery / OpenCASCADE shape.

Every requirement predicate reads its evidence from `ShapeAnalysis`:
bounding box, volume, topology counts, genus, cylindrical features (holes
and bosses), planar faces, point classification and symmetry checks.
Expensive quantities are computed lazily and cached per shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Optional, Sequence

import numpy as np
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp_Face
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from OCP.gp import gp_Pnt, gp_Vec

TWO_PI = 2.0 * math.pi
FULL_SPAN_FRACTION = 0.95  # a cylindrical feature spanning >= 95% of 2*pi counts as full


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

AXIS_NAMES = {
    "X": (1.0, 0.0, 0.0), "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0), "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "Z": (0.0, 0.0, 1.0), "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}


def parse_direction(value) -> np.ndarray:
    """Parse "X", "+Z", "-Y" or a 3-vector into a unit numpy vector."""
    if isinstance(value, str):
        key = value.strip().upper()
        if key not in AXIS_NAMES:
            raise ValueError(f"Unknown axis {value!r}")
        return np.array(AXIS_NAMES[key], dtype=float)
    v = np.asarray(value, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero direction vector")
    return v / n


def canonical_axis(d: np.ndarray) -> np.ndarray:
    """Sign-normalise an axis so that parallel and anti-parallel axes compare equal."""
    d = d / np.linalg.norm(d)
    i = int(np.argmax(np.abs(d)))
    return d if d[i] > 0 else -d


def axis_point_near_origin(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Point on the line (p, d) closest to the origin."""
    return p - np.dot(p, d) * d


def point_line_distance(q: np.ndarray, p: np.ndarray, d: np.ndarray) -> float:
    """Distance from q to the infinite line through p with unit direction d."""
    w = q - p
    return float(np.linalg.norm(w - np.dot(w, d) * d))


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis through the origin."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def as_shape(obj) -> Optional[cq.Shape]:
    """Return a single cq.Shape for a Workplane, Shape or list of shapes."""
    if obj is None:
        return None
    if isinstance(obj, cq.Workplane):
        shapes = [o for o in obj.objects if isinstance(o, cq.Shape)]
        solids = [s for s in shapes if s.ShapeType() in ("Solid", "Compound", "CompSolid")]
        if not solids:
            return None
        return solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    if isinstance(obj, cq.Shape):
        return obj
    if isinstance(obj, (list, tuple)):
        shapes = [s for s in obj if isinstance(s, cq.Shape)]
        if not shapes:
            return None
        return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
    return None


# ---------------------------------------------------------------------------
# Feature records
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class CylFeature:
    """A cylindrical feature: one or more coaxial cylinder faces of equal radius."""
    radius: float
    axis: np.ndarray            # canonical unit direction
    point: np.ndarray           # point on the axis closest to the origin
    t_min: float                # axial extent, measured along `axis`
    t_max: float
    span: float                 # summed angular span of the faces (radians)
    concave: bool               # True: hole / bore; False: boss / shaft
    n_faces: int = 1
    through: Optional[bool] = None

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def full(self) -> bool:
        return self.span >= FULL_SPAN_FRACTION * TWO_PI

    @property
    def depth(self) -> float:
        return self.t_max - self.t_min

    def center_near(self, target: np.ndarray) -> np.ndarray:
        """Point on the axis closest to `target` (for position checks)."""
        return self.point + np.dot(target - self.point, self.axis) * self.axis

    def to_dict(self) -> dict:
        return {
            "kind": "hole" if self.concave else "boss",
            "diameter": round(self.diameter, 4),
            "axis": [round(float(x), 4) for x in self.axis],
            "axis_point": [round(float(x), 3) for x in self.point],
            "t_range": [round(self.t_min, 3), round(self.t_max, 3)],
            "depth": round(self.depth, 3),
            "full": self.full,
            "span_deg": round(math.degrees(self.span), 1),
            "through": self.through,
        }


@dataclass(eq=False)
class PlaneFace:
    normal: np.ndarray   # outward unit normal
    offset: float        # normal . point
    area: float
    center: np.ndarray

    def to_dict(self) -> dict:
        return {
            "normal": [round(float(x), 4) for x in self.normal],
            "offset": round(self.offset, 4),
            "area": round(self.area, 3),
            "center": [round(float(x), 3) for x in self.center],
        }


@dataclass
class Topology:
    n_solids: int
    n_shells: int
    n_faces: int
    n_wires: int
    n_edges: int
    n_vertices: int
    genus: int                       # total number of through-handles (Euler-Poincare)
    face_types: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class ShapeAnalysis:
    """Lazy, cached kernel measurements of one shape."""

    def __init__(self, shape: cq.Shape, sample_seed: int = 0):
        if shape is None:
            raise ValueError("ShapeAnalysis needs a shape")
        self.shape = shape
        self._seed = sample_seed

    # --- global quantities -------------------------------------------------

    @cached_property
    def bbox(self) -> dict:
        # exact geometry, ignoring any tessellation and shape tolerances (CadQuery's default can
        # use a mesh once the shape has been tessellated, which inflates boxes by ~0.02 mm)
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        box = Bnd_Box()
        try:
            BRepBndLib.AddOptimal_s(self.shape.wrapped, box, False, False)
            bb = cq.occ_impl.geom.BoundBox(box)
        except Exception:
            bb = self.shape.BoundingBox()
        return {
            "xmin": bb.xmin, "xmax": bb.xmax, "xlen": bb.xlen,
            "ymin": bb.ymin, "ymax": bb.ymax, "ylen": bb.ylen,
            "zmin": bb.zmin, "zmax": bb.zmax, "zlen": bb.zlen,
        }

    @cached_property
    def extent(self) -> float:
        b = self.bbox
        return max(b["xlen"], b["ylen"], b["zlen"], 1e-9)

    @cached_property
    def volume(self) -> float:
        try:
            return float(self.shape.Volume())
        except Exception:
            return 0.0

    @cached_property
    def area(self) -> float:
        try:
            return float(self.shape.Area())
        except Exception:
            return 0.0

    @cached_property
    def center_of_mass(self) -> np.ndarray:
        try:
            return np.array(self.shape.Center().toTuple(), dtype=float)
        except Exception:
            b = self.bbox
            return np.array([(b["xmin"] + b["xmax"]) / 2, (b["ymin"] + b["ymax"]) / 2, (b["zmin"] + b["zmax"]) / 2])

    @cached_property
    def is_valid(self) -> bool:
        try:
            return bool(self.shape.isValid())
        except Exception:
            return False

    @cached_property
    def solids(self) -> list:
        try:
            return list(self.shape.Solids())
        except Exception:
            return []

    @cached_property
    def topology(self) -> Topology:
        s = self.shape
        faces = s.Faces()
        face_types: dict[str, int] = {}
        for f in faces:
            t = f.geomType()
            face_types[t] = face_types.get(t, 0) + 1
        genus_total = 0
        for solid in (self.solids or [s]):
            v, e, f = len(solid.Vertices()), len(solid.Edges()), len(solid.Faces())
            w, sh = len(solid.Wires()), max(len(solid.Shells()), 1)
            rings = max(w - f, 0)
            chi = v - e + f - rings           # = 2 (S - G)
            genus_total += max(int(round(sh - chi / 2.0)), 0)
        return Topology(
            n_solids=len(self.solids),
            n_shells=len(s.Shells()),
            n_faces=len(faces),
            n_wires=len(s.Wires()),
            n_edges=len(s.Edges()),
            n_vertices=len(s.Vertices()),
            genus=genus_total,
            face_types=face_types,
        )

    # --- faces -------------------------------------------------------------

    @staticmethod
    def _face_point_normal(face: cq.Face, u: float, v: float):
        p, n = gp_Pnt(), gp_Vec()
        BRepGProp_Face(face.wrapped).Normal(u, v, p, n)
        nv = np.array([n.X(), n.Y(), n.Z()], dtype=float)
        nn = np.linalg.norm(nv)
        return np.array([p.X(), p.Y(), p.Z()], dtype=float), (nv / nn if nn > 0 else nv)

    @cached_property
    def cylinders(self) -> list[CylFeature]:
        """Cylindrical features, merged across split faces, with concavity and through-ness."""
        feats: list[CylFeature] = []
        for face in self.shape.Faces():
            if face.geomType() != "CYLINDER":
                continue
            try:
                cyl = BRepAdaptor_Surface(face.wrapped).Cylinder()
                umin, umax, vmin, vmax = face._uvBounds()
            except Exception:
                continue
            r = float(cyl.Radius())
            ax = cyl.Axis()
            loc = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
            dvec = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
            d = canonical_axis(dvec)
            p0 = axis_point_near_origin(loc, d)
            # axial extent: v is measured along the cylinder's own direction
            t_vals = [float(np.dot(loc + vv * dvec, d)) for vv in (vmin, vmax)]
            span = min(float(umax - umin), TWO_PI)
            pm, nm = self._face_point_normal(face, 0.5 * (umin + umax), 0.5 * (vmin + vmax))
            w = pm - loc
            radial = w - np.dot(w, dvec) * dvec
            concave = bool(np.dot(nm, radial) < 0)
            merged = False
            for ft in feats:
                if (ft.concave == concave and abs(ft.radius - r) < 1e-4
                        and np.linalg.norm(np.cross(ft.axis, d)) < 1e-4
                        and np.linalg.norm(ft.point - p0) < 1e-3):
                    ft.span = min(ft.span + span, TWO_PI)
                    ft.t_min = min(ft.t_min, *t_vals)
                    ft.t_max = max(ft.t_max, *t_vals)
                    ft.n_faces += 1
                    merged = True
                    break
            if not merged:
                feats.append(CylFeature(radius=r, axis=d, point=p0, t_min=min(t_vals), t_max=max(t_vals),
                                        span=span, concave=concave))
        # through-ness for full holes: the axis leaves into empty space at both ends
        for ft in feats:
            if ft.concave and ft.full:
                eps = max(0.05 * ft.depth, 0.2)
                a = ft.point + (ft.t_min - eps) * ft.axis
                b = ft.point + (ft.t_max + eps) * ft.axis
                ft.through = (not self.contains(a)) and (not self.contains(b))
        return feats

    def holes(self, full_only: bool = True) -> list[CylFeature]:
        return [c for c in self.cylinders if c.concave and (c.full or not full_only)]

    def bosses(self, full_only: bool = True) -> list[CylFeature]:
        return [c for c in self.cylinders if (not c.concave) and (c.full or not full_only)]

    @cached_property
    def planes(self) -> list[PlaneFace]:
        out = []
        for face in self.shape.Faces():
            if face.geomType() != "PLANE":
                continue
            try:
                umin, umax, vmin, vmax = face._uvBounds()
                p, n = self._face_point_normal(face, 0.5 * (umin + umax), 0.5 * (vmin + vmax))
                c = np.array(face.Center().toTuple(), dtype=float)
                out.append(PlaneFace(normal=n, offset=float(np.dot(n, c)), area=float(face.Area()), center=c))
            except Exception:
                continue
        return out

    # --- point queries -------------------------------------------------------

    def contains(self, point: Sequence[float], tol: float = 1e-6) -> bool:
        """True if the point lies inside (or on) any solid of the shape."""
        p = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))
        targets = self.solids or [self.shape]
        for s in targets:
            try:
                state = BRepClass3d_SolidClassifier(s.wrapped, p, tol).State()
            except Exception:
                continue
            if state in (TopAbs_IN, TopAbs_ON):
                return True
        return False

    # --- surface sampling and symmetry ----------------------------------------

    @cached_property
    def _mesh(self):
        tol = max(self.extent * 1e-3, 1e-3)
        verts, tris = self.shape.tessellate(tol, 0.2)
        v = np.array([x.toTuple() for x in verts], dtype=float)
        t = np.array(tris, dtype=np.int64).reshape(-1, 3) if len(tris) else np.zeros((0, 3), dtype=np.int64)
        return v, t

    def sample_surface(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        v, t = self._mesh
        if len(t) == 0:
            return v.copy()
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        total = areas.sum()
        if total <= 0:
            return v.copy()
        rng = np.random.default_rng(self._seed if seed is None else seed)
        idx = rng.choice(len(t), size=n, p=areas / total)
        r1, r2 = rng.random(n), rng.random(n)
        s1 = np.sqrt(r1)
        return (1 - s1)[:, None] * a[idx] + (s1 * (1 - r2))[:, None] * b[idx] + (s1 * r2)[:, None] * c[idx]

    @cached_property
    def _trimesh(self):
        import trimesh
        v, t = self._mesh
        return trimesh.Trimesh(v, t, process=False) if len(t) else None

    def distance_to_surface(self, points: np.ndarray) -> np.ndarray:
        """Exact distance from each point to the tessellated surface (trimesh + rtree)."""
        import trimesh
        m = self._trimesh
        if m is None:
            return np.full(len(points), np.inf)
        _, d, _ = trimesh.proximity.closest_point(m, points)
        return np.asarray(d, dtype=float)

    def transform_deviation(self, matrix: np.ndarray, offset: np.ndarray, n: int = 600) -> float:
        """99th-percentile distance between the transformed surface samples and the original surface."""
        pts = self.sample_surface(n, seed=self._seed + 2)
        moved = pts @ matrix.T + offset
        return float(np.quantile(self.distance_to_surface(moved), 0.99))

    @property
    def tessellation_tol(self) -> float:
        return max(self.extent * 1e-3, 1e-3)

    def symmetry_threshold(self, tol: Optional[float] = None) -> float:
        base = max(3.0 * self.tessellation_tol, 0.02)
        return max(base, tol or 0.0)

    def features_map_onto_themselves(self, matrix: np.ndarray, offset: np.ndarray, tol: float) -> bool:
        """Every complete cylindrical feature maps onto one of the same radius and kind.

        Surface sampling (99th percentile) cannot see a small feature such as a tapped hole moving,
        because it covers well under 1 % of the surface; this exact check does."""
        feats = [f for f in self.cylinders if f.full]
        for f in feats:
            p = f.point @ matrix.T + offset
            a = canonical_axis(f.axis @ matrix.T)
            if not any(g.concave == f.concave and abs(g.radius - f.radius) <= tol
                       and np.linalg.norm(np.cross(g.axis, a)) < 1e-3
                       and point_line_distance(p, g.point, g.axis) <= max(tol, 1e-3) for g in feats):
                return False
        return True

    def is_rotation_symmetric(self, axis, center, angle: float, tol: Optional[float] = None) -> tuple[bool, float]:
        d = parse_direction(axis)
        c = np.asarray(center, dtype=float)
        R = rotation_matrix(d, angle)
        dev = self.transform_deviation(R, c - R @ c)
        thr = self.symmetry_threshold(tol)
        return dev <= thr and self.features_map_onto_themselves(R, c - R @ c, thr), dev

    def is_reflection_symmetric(self, normal, offset: float = 0.0, tol: Optional[float] = None) -> tuple[bool, float]:
        nvec = parse_direction(normal)
        H = np.eye(3) - 2.0 * np.outer(nvec, nvec)
        shift = 2.0 * offset * nvec
        dev = self.transform_deviation(H, shift)
        thr = self.symmetry_threshold(tol)
        return dev <= thr and self.features_map_onto_themselves(H, shift, thr), dev

    # --- compact summary for prompts and logs --------------------------------

    def summary(self, max_features: int = 12) -> dict:
        topo = self.topology
        holes = sorted(self.holes(full_only=False), key=lambda c: -c.diameter)
        bosses = sorted(self.bosses(full_only=True), key=lambda c: -c.diameter)
        return {
            "valid": self.is_valid,
            "n_solids": topo.n_solids,
            "bbox": {k: round(v, 3) for k, v in self.bbox.items()},
            "volume": round(self.volume, 3),
            "area": round(self.area, 3),
            "center_of_mass": [round(float(x), 3) for x in self.center_of_mass],
            "genus": topo.genus,
            "faces": topo.n_faces,
            "face_types": topo.face_types,
            "holes": [h.to_dict() for h in holes[:max_features]],
            "n_holes_full": sum(1 for h in holes if h.full),
            "bosses": [b.to_dict() for b in bosses[:max_features]],
        }


def analyse(obj) -> Optional[ShapeAnalysis]:
    shape = as_shape(obj)
    return ShapeAnalysis(shape) if shape is not None else None
