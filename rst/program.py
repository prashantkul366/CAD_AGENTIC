"""Statement-level view of a CadQuery program.

The unit of localisation and repair is a top-level statement. This module
splits a program into statements, maps source lines to statements, computes
simple def-use dependencies, and splices replacement code back in.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", re.S)


def strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if any."""
    t = text.strip()
    m = FENCE.match(t)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


@dataclass
class Statement:
    idx: int
    start: int          # 1-based first line
    end: int            # 1-based last line (inclusive)
    code: str
    defines: set = field(default_factory=set)
    uses: set = field(default_factory=set)
    is_import: bool = False
    is_param: bool = False      # a plain value assignment (no calls), e.g. `height = 1.5`

    def label(self) -> str:
        return f"S{self.idx}"


class Program:
    """A parsed program: source plus its top-level statements."""

    def __init__(self, source: str):
        self.source = source.replace("\r\n", "\n")
        self.lines = self.source.split("\n")
        self.tree = ast.parse(self.source)
        self.statements: list[Statement] = []
        for i, node in enumerate(self.tree.body):
            start = node.lineno
            if getattr(node, "decorator_list", None):
                start = min(d.lineno for d in node.decorator_list)
            end = node.end_lineno or node.lineno
            code = "\n".join(self.lines[start - 1:end])
            st = Statement(idx=i, start=start, end=end, code=code,
                           is_import=isinstance(node, (ast.Import, ast.ImportFrom)))
            st.defines, st.uses = _def_use(node)
            st.is_param = (isinstance(node, (ast.Assign, ast.AnnAssign))
                           and getattr(node, "value", None) is not None
                           and not any(isinstance(n, ast.Call) for n in ast.walk(node.value)))
            self.statements.append(st)

    # --- lookups ------------------------------------------------------------

    def statement_at_line(self, lineno: Optional[int]) -> Optional[int]:
        if lineno is None:
            return None
        for st in self.statements:
            if st.start <= lineno <= st.end:
                return st.idx
        return None

    def dependencies(self, idx: int, max_extra: int = 2) -> list[int]:
        """Statements (before idx) that define auxiliary names used by statement idx, nearest first.

        Names the statement itself reassigns (the main variable in
        `result = result.op(...)`) are excluded: their previous definition is
        the preceding construction step, not a tool body.
        """
        st = self.statements[idx]
        bodies, params = [], []
        for name in sorted(st.uses - st.defines):
            for j in range(idx - 1, -1, -1):
                other = self.statements[j]
                if name in other.defines and not other.is_import:
                    target = params if other.is_param else bodies
                    if j not in target:
                        target.append(j)
                    break
        # plain parameter definitions are cheap to include in full (transitively: radius = diameter / 2
        # pulls in diameter); body definitions are capped
        todo = list(params)
        seen = set(params)
        while todo:
            j = todo.pop()
            for name in self.statements[j].uses - self.statements[j].defines:
                for k in range(j - 1, -1, -1):
                    other = self.statements[k]
                    if name in other.defines:
                        if other.is_param and k not in seen:
                            seen.add(k)
                            todo.append(k)
                        break
        bodies.sort(reverse=True)
        return sorted(set(bodies[:max_extra]) | seen, reverse=True)

    def numbered(self, highlight: Optional[set] = None) -> str:
        """Program text with a statement label before each top-level statement."""
        out = []
        for st in self.statements:
            mark = "  <<< EDIT" if highlight and st.idx in highlight else ""
            out.append(f"# [{st.label()}]{mark}\n{st.code}")
        return "\n".join(out)

    # --- editing --------------------------------------------------------------

    def replace(self, edits: dict[int, str], insert_after: Optional[dict[int, str]] = None) -> str:
        """Return new source with statements replaced (and optional insertions after statements)."""
        insert_after = insert_after or {}
        pieces = []
        # keep any leading comments / blank lines before the first statement
        first = self.statements[0].start if self.statements else len(self.lines) + 1
        if first > 1:
            pieces.append("\n".join(self.lines[:first - 1]))
        for st in self.statements:
            code = edits.get(st.idx, st.code)
            if code is not None and code.strip():
                pieces.append(code.rstrip())
            if st.idx in insert_after and insert_after[st.idx].strip():
                pieces.append(insert_after[st.idx].rstrip())
        if -1 in insert_after:  # insert before everything (after imports is safer; caller decides)
            pieces.insert(0, insert_after[-1].rstrip())
        return "\n".join(pieces) + "\n"


def _def_use(node) -> tuple[set, set]:
    defines, uses = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Store):
                defines.add(n.id)
            elif isinstance(n.ctx, ast.Load):
                uses.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defines.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                defines.add((a.asname or a.name).split(".")[0])
    # names assigned via augmented assignment are both used and defined
    return defines, uses - {"cq", "math", "np", "numpy", "cadquery"}


def parses(code: str) -> Optional[str]:
    """None if the code parses, else the SyntaxError message."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
