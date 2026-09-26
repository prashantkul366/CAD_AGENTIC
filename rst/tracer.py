"""Feature tracer: record every intermediate solid of a CadQuery program.

CadQuery builds a new `Workplane` for every operation through
`Workplane.newObject`. We hook that method while the user program runs and
record an event whenever an operation produces a solid that differs from
the solid in its parent's context. After execution the events on the
lineage of the final `result` (its chain of parents) form the construction
timeline S_1 ... S_T, each mapped to the source line and CadQuery method that
produced it. One execution yields every prefix state.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import cadquery as cq

from .geometry import as_shape

RESULT_NAMES = ["result", "model", "part", "shape", "assembly", "drone", "chassis", "bracket", "plate", "body"]


def _solids(wp) -> list:
    if not isinstance(wp, cq.Workplane):
        return []
    return [o for o in wp.objects if isinstance(o, cq.Shape) and o.ShapeType() in ("Solid", "Compound", "CompSolid")]


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x.wrapped.IsSame(y.wrapped) for x, y in zip(a, b))


def _context_solids(wp) -> list:
    w = wp
    while w is not None:
        s = _solids(w)
        if s:
            return s
        w = getattr(w, "parent", None)
    return []


@dataclass
class Event:
    order: int
    wp: object                 # the new Workplane (kept alive for identity checks)
    shape: object              # cq.Shape of the new solid
    op: str                    # CadQuery method called from user code
    lineno: Optional[int]      # outermost user-code line (top-level statement)
    inner_lineno: Optional[int]  # innermost user-code line (e.g. inside a helper function)


class FeatureTracer:
    def __init__(self, script_path: str):
        self.script_norm = os.path.normcase(os.path.abspath(script_path))
        self.events: list[Event] = []
        self._orig = None

    # --- patching -----------------------------------------------------------

    def __enter__(self):
        tracer = self
        orig = cq.Workplane.newObject
        self._orig = orig

        def newObject(wp_self, objlist):
            new = orig(wp_self, objlist)
            try:
                tracer._record(wp_self, new)
            except Exception:
                pass
            return new

        cq.Workplane.newObject = newObject
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            cq.Workplane.newObject = self._orig
        return False

    # --- recording --------------------------------------------------------------

    def _user_lines(self):
        """(outermost, innermost) user-code line numbers and the CadQuery method called from user code."""
        f = sys._getframe(3)
        inner = outer = None
        op = "?"
        prev = None
        while f is not None:
            if os.path.normcase(os.path.abspath(f.f_code.co_filename)) == self.script_norm:
                if inner is None:
                    inner = f.f_lineno
                    op = prev.f_code.co_name if prev is not None else "?"
                outer = f.f_lineno
            prev = f
            f = f.f_back
        return outer, inner, op

    def _record(self, parent, new):
        solids = _solids(new)
        if not solids:
            return
        if _same(_context_solids(parent), solids):
            return
        outer, inner, op = self._user_lines()
        shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
        self.events.append(Event(len(self.events), new, shape, op, outer, inner))

    # --- lineage --------------------------------------------------------------------

    def timeline(self, result) -> list[Event]:
        """Events on the parent chain of `result`, in execution order."""
        if not isinstance(result, cq.Workplane):
            return []
        chain = set()
        root = None
        w = result
        while w is not None:
            chain.add(id(w))
            root = w
            w = getattr(w, "parent", None)
        tl = [e for e in self.events if id(e.wp) in chain]
        root_solids = _solids(root)
        if root_solids and not any(e.wp is root for e in tl):
            shape = root_solids[0] if len(root_solids) == 1 else cq.Compound.makeCompound(root_solids)
            tl.insert(0, Event(-1, root, shape, "init", None, None))
        return tl


def find_result(namespace: dict):
    """Locate the final shape in the executed namespace (same rule as the CADSmith executor)."""
    for name in RESULT_NAMES:
        obj = namespace.get(name)
        if isinstance(obj, cq.Workplane):
            return obj
    for name, obj in reversed(list(namespace.items())):
        if not name.startswith("_") and isinstance(obj, cq.Workplane):
            return obj
    for name in RESULT_NAMES:
        obj = namespace.get(name)
        if isinstance(obj, cq.Shape):
            return obj
    return None


def final_shape(result):
    if isinstance(result, cq.Shape):
        return result
    s = as_shape(result)
    if s is not None:
        return s
    ctx = _context_solids(result)
    if ctx:
        return ctx[0] if len(ctx) == 1 else cq.Compound.makeCompound(ctx)
    return None
