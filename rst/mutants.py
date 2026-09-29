"""Fault injection for CadQuery programs (attribution and robustness experiments).

Each mutant changes exactly one top-level statement, so the ground-truth
culprit is known. Operators:

  param_shift      a size literal (length, radius, depth, ...) scaled beyond tolerance
  placement_shift  a position literal (center/translate/pushPoints/offset) moved
  count_change     a count literal (polarArray/rarray/polygon/range) changed by +/-1
  wrong_workplane  a face selector or plane name flipped (">Z" -> "<Z", "XY" -> "XZ")
  feature_delete   a `result = result.<op>(...)` statement replaced by `result = result`
"""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass
from typing import Optional

POSITION_CALLS = {"center", "translate", "moveTo", "move", "pushPoints", "transformed", "workplane", "moved",
                  "located", "rotate", "rotateAboutCenter"}
COUNT_ARGS = {"polarArray": 3, "rarray": (2, 3), "polygon": 0, "range": 0, "regularPolygon": 1}
SELECTOR_FLIPS = {">Z": "<Z", "<Z": ">Z", ">X": "<X", "<X": ">X", ">Y": "<Y", "<Y": ">Y",
                  "XY": "XZ", "XZ": "YZ", "YZ": "XY", "|Z": "|X", "|X": "|Y", "|Y": "|Z"}


@dataclass
class Mutant:
    kind: str
    stmt: int
    code: str
    detail: str


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _sites(tree: ast.Module):
    """Yield (kind, stmt_idx, locator) for every mutable site."""
    for si, stmt in enumerate(tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        parents = {}
        for node in ast.walk(stmt):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(stmt):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                call, argpos = _enclosing_call(node, parents)
                name = _call_name(call) if call is not None else ""
                if name in COUNT_ARGS:
                    pos = COUNT_ARGS[name]
                    pos = pos if isinstance(pos, tuple) else (pos,)
                    if argpos in pos and isinstance(node.value, int) and node.value >= 2:
                        yield ("count_change", si, node)
                        continue
                if abs(node.value) < 0.5:
                    continue
                if name in POSITION_CALLS:
                    yield ("placement_shift", si, node)
                elif isinstance(parents.get(node), ast.Subscript):
                    continue
                else:
                    yield ("param_shift", si, node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in SELECTOR_FLIPS:
                yield ("wrong_workplane", si, node)
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call) and si > 1):
            target = stmt.targets[0].id
            root = stmt.value
            while isinstance(root, ast.Call) and isinstance(root.func, ast.Attribute):
                root = root.func.value
            if isinstance(root, ast.Name) and root.id == target:
                yield ("feature_delete", si, None)


def _enclosing_call(node, parents):
    child, cur = node, parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.Call):
            if child in cur.args:
                return cur, cur.args.index(child)
            for kw in cur.keywords:
                if kw.value is child:
                    return cur, kw.arg
            return cur, None
        if isinstance(cur, (ast.Tuple, ast.List, ast.UnaryOp, ast.BinOp)):
            child, cur = cur, parents.get(cur)
            continue
        return None, None
    return None, None


def _apply(tree: ast.Module, kind: str, si: int, node, rng: random.Random, extent: float) -> Optional[str]:
    t = copy.deepcopy(tree)
    if kind == "feature_delete":
        target = t.body[si].targets[0].id
        t.body[si] = ast.parse(f"{target} = {target}").body[0]
        return ast.unparse(t)
    # locate the same node in the copy by walking in the same order
    orig_nodes = list(ast.walk(tree.body[si]))
    idx = orig_nodes.index(node)
    new_node = list(ast.walk(t.body[si]))[idx]
    v = node.value
    if kind == "param_shift":
        factor = rng.choice([0.6, 0.75, 1.3, 1.5])
        nv = v * factor
        if abs(nv - v) < 0.5:
            nv = v + (1.0 if factor > 1 else -1.0)
        new_node.value = type(v)(round(nv, 3)) if isinstance(v, float) else int(round(nv)) or v + 1
    elif kind == "placement_shift":
        delta = max(2.0, round(0.1 * extent, 1)) * rng.choice([-1, 1])
        new_node.value = (v + delta) if isinstance(v, float) else int(round(v + delta))
    elif kind == "count_change":
        new_node.value = v + 1 if v < 3 or rng.random() < 0.5 else v - 1
    elif kind == "wrong_workplane":
        new_node.value = SELECTOR_FLIPS[v]
    return ast.unparse(t)


def make_mutants(code: str, n: int, seed: int = 0, extent: float = 50.0,
                 kinds: Optional[set] = None) -> list[Mutant]:
    """Up to n distinct single-statement mutants, spread over operator kinds."""
    tree = ast.parse(code)
    sites = list(_sites(tree))
    if kinds:
        sites = [s for s in sites if s[0] in kinds]
    rng = random.Random(seed)
    by_kind: dict[str, list] = {}
    for s in sites:
        by_kind.setdefault(s[0], []).append(s)
    for v in by_kind.values():
        rng.shuffle(v)
    out, seen = [], set()
    order = sorted(by_kind)
    while len(out) < n and any(by_kind.values()):
        for k in order:
            if not by_kind[k] or len(out) >= n:
                continue
            kind, si, node = by_kind[k].pop()
            new = _apply(tree, kind, si, node, rng, extent)
            if new is None or new in seen:
                continue
            seen.add(new)
            detail = kind if node is None else f"{kind}: {getattr(node, 'value', '')!r}"
            out.append(Mutant(kind, si, new, detail))
    return out


def normalised(code: str) -> str:
    """The unmutated program rendered the same way mutants are (ast.unparse)."""
    return ast.unparse(ast.parse(code))
