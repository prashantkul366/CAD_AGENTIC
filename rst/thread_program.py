"""Rewrite fluent CadQuery chains into state-threaded form (one feature per statement).

    result = cq.Workplane("XY").box(10, 10, 2).faces(">Z").workplane().hole(3).edges("|Z").fillet(1)
becomes
    result = cq.Workplane("XY").box(10, 10, 2)
    result = result.faces(">Z").workplane().hole(3)
    result = result.edges("|Z").fillet(1)

Splitting happens only right after solid-producing calls, where the
intermediate Workplane holds the solid, so behaviour is unchanged (the
experiment script verifies volume and bounding box). Statements that reuse the
target name inside the chain are left alone.
"""

from __future__ import annotations

import ast

SOLID_OPS = {
    "box", "cylinder", "sphere", "wedge", "extrude", "twistExtrude", "revolve", "loft", "sweep", "cutBlind",
    "cutThruAll", "hole", "cboreHole", "cskHole", "union", "cut", "intersect", "fillet", "chamfer", "shell",
    "mirror", "split", "combine", "text", "translate", "rotate", "rotateAboutCenter", "clean",
}


def _links(expr):
    """Decompose a.b(...).c(...) into (base, [call nodes from innermost to outermost])."""
    calls = []
    node = expr
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        calls.append(node)
        node = node.func.value
    calls.reverse()
    return node, calls


def _uses_name(node, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def thread_statement(stmt: ast.stmt) -> list[ast.stmt]:
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
        return [stmt]
    target = stmt.targets[0].id
    base, calls = _links(stmt.value)
    if len(calls) < 2:
        return [stmt]
    split_after = [i for i, c in enumerate(calls[:-1]) if c.func.attr in SOLID_OPS]
    if not split_after:
        return [stmt]
    # do not split if the target name is read inside the chain arguments after the first split point
    first = split_after[0]
    for c in calls[first + 1:]:
        if any(_uses_name(a, target) for a in list(c.args) + [k.value for k in c.keywords]):
            return [stmt]
    out = []
    start = 0
    cur_base = base
    for cut in split_after + [len(calls) - 1]:
        expr = cur_base
        for c in calls[start:cut + 1]:
            expr = ast.Call(func=ast.Attribute(value=expr, attr=c.func.attr, ctx=ast.Load()), args=c.args,
                            keywords=c.keywords)
        out.append(ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=expr, lineno=0))
        cur_base = ast.Name(id=target, ctx=ast.Load())
        start = cut + 1
    return out


def thread_program(code: str) -> tuple[str, int]:
    """Return (threaded code, number of statements added)."""
    tree = ast.parse(code)
    body, added = [], 0
    for stmt in tree.body:
        new = thread_statement(stmt)
        added += len(new) - 1
        body.extend(new)
    tree.body = body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", added
