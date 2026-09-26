"""Core tests for the RST kernel layer: geometry, predicates, tracer, matrix, blame, repair loop."""

import json
import math

import cadquery as cq
import pytest

from rst.geometry import analyse
from rst.kernel import Kernel
from rst.llm import ScriptedLLM
from rst.localize import localize
from rst.matrix import Trajectory
from rst.pipeline import Engine, RSTConfig, run_rst, run_tests_log
from rst.program import Program
from rst.requirements import CATALOGUE, Requirement, evaluate, load_requirements, validate


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,wp,genus", [
    ("box", lambda: cq.Workplane().box(50, 30, 20), 0),
    ("plate4", lambda: cq.Workplane().box(80, 60, 4).faces(">Z").workplane()
        .rect(70, 50, forConstruction=True).vertices().hole(3.4), 4),
    ("tube", lambda: cq.Workplane().circle(10).circle(6).extrude(20), 1),
    ("torus", lambda: cq.Workplane().add(cq.Solid.makeTorus(20, 5)), 1),
    ("sphere", lambda: cq.Workplane().sphere(10), 0),
    ("cone", lambda: cq.Workplane().add(cq.Solid.makeCone(10, 0, 20)), 0),
    ("blind", lambda: cq.Workplane().box(40, 40, 20).faces(">Z").workplane().hole(8, depth=10), 0),
])
def test_genus(name, wp, genus):
    assert analyse(wp()).topology.genus == genus


def test_hole_classification():
    A = analyse(cq.Workplane().box(40, 40, 20).faces(">Z").workplane().hole(8, depth=10))
    (h,) = A.holes(True)
    assert h.concave and h.full and h.through is False and abs(h.diameter - 8) < 1e-6
    B = analyse(cq.Workplane().box(40, 40, 4).faces(">Z").workplane().center(0, 20).hole(6))
    assert B.holes(True) == [] and len(B.holes(False)) == 1  # half hole cut through an edge


def test_symmetry():
    gear = cq.Workplane().circle(20).extrude(5).union(cq.Workplane().polarArray(21, 0, 360, 16).rect(3, 4).extrude(5))
    A = analyse(gear)
    assert A.is_rotation_symmetric("Z", [0, 0, 0], 2 * math.pi / 16)[0]
    assert not A.is_rotation_symmetric("Z", [0, 0, 0], 2 * math.pi / 15)[0]
    assert not A.is_rotation_symmetric("Z", [0, 0, 0], 2 * math.pi / 32)[0]
    off = analyse(cq.Workplane().box(40, 20, 10).faces(">Z").workplane().center(5, 0).hole(6))
    assert off.is_reflection_symmetric("Y", 0)[0] and not off.is_reflection_symmetric("X", 0)[0]


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------

def _plate():
    return analyse(cq.Workplane("XY").box(80, 60, 4, centered=(True, True, False)).faces(">Z").workplane()
                   .rect(70, 50, forConstruction=True).vertices().hole(3.4))


@pytest.mark.parametrize("req,expected", [
    ({"type": "bbox_size", "params": {"x": 80, "y": 60, "z": 4}}, True),
    ({"type": "bbox_size", "params": {"x": 60, "y": 80, "z": 4, "any_orientation": True}}, True),
    ({"type": "bbox_size", "params": {"x": 80, "y": 60, "z": 5}}, False),
    ({"type": "bbox_range", "params": {"axis": "Z", "min": 0, "max": 4}}, True),
    ({"type": "hole_count", "params": {"count": 4, "diameter": 3.4, "through": True}}, True),
    ({"type": "hole_count", "params": {"count": 4, "diameter": 3.0}}, False),
    ({"type": "hole_at", "params": {"center": [35, 25, 0], "diameter": 3.4, "axis": "Z"}}, True),
    ({"type": "hole_at", "params": {"center": [30, 25, 0], "diameter": 3.4, "axis": "Z"}}, False),
    ({"type": "through_hole_count", "params": {"count": 4}}, True),
    ({"type": "planar_face_at", "params": {"normal": "+Z", "offset": 4}}, True),
    ({"type": "planar_face_at", "params": {"normal": "-Z", "offset": 0}}, True),
    ({"type": "material_at", "params": {"point": [0, 0, 2], "present": True}}, True),
    ({"type": "material_at", "params": {"point": [35, 25, 2], "present": False}}, True),
    ({"type": "reflection_symmetry", "params": {"normal": "X"}}, True),
    ({"type": "volume", "params": {"value": 80 * 60 * 4 - 4 * math.pi * 1.7 ** 2 * 4, "rel_tol": 0.001}}, True),
    ({"type": "single_solid"}, True),
    ({"type": "valid"}, True),
])
def test_predicates(req, expected):
    r = Requirement.from_dict({"id": "r", **req})
    assert validate(r) is None
    assert evaluate(r, _plate()).passed is expected


def test_bolt_circle_and_coaxial():
    flange = (cq.Workplane().circle(25).extrude(10).faces(">Z").workplane().hole(14)
              .faces(">Z").workplane().polarArray(19, 0, 360, 6).hole(6.5))
    A = analyse(flange)
    ok = Requirement.from_dict({"id": "b", "type": "bolt_circle", "params": {"count": 6, "diameter": 6.5, "pitch_diameter": 38}})
    bad = Requirement.from_dict({"id": "b", "type": "bolt_circle", "params": {"count": 8, "diameter": 6.5, "pitch_diameter": 38}})
    co = Requirement.from_dict({"id": "c", "type": "coaxial", "params": {"diameters": [50, 14]}})
    assert evaluate(ok, A).passed and not evaluate(bad, A).passed and evaluate(co, A).passed


def test_catalogue_examples_validate():
    for name, spec in CATALOGUE.items():
        assert validate(Requirement(id="x", type=name, params=spec.example)) is None, name


def test_load_requirements_rejects_bad_items():
    reqs, errors = load_requirements([{"type": "hole_at", "params": {"diameter": 3}}, {"type": "nope"}, {"type": "valid"}])
    assert len(reqs) == 1 and len(errors) == 2


# ---------------------------------------------------------------------------
# program + tracer + matrix + blame
# ---------------------------------------------------------------------------

def test_program_dependencies_exclude_main_variable():
    prog = Program("import cadquery as cq\nresult = cq.Workplane().box(1,1,1)\nwall = cq.Workplane().box(1,2,3)\n"
                   "result = result.union(wall)\n")
    assert prog.dependencies(3) == [2]


PLATE_PROMPT = "Plate 80 x 60 x 4 mm on XY from Z=0 with four 3.4 mm through-holes at (+-35, +-25) and a 20 mm centre bore."
PLATE_BUGGY = '''import cadquery as cq
result = cq.Workplane("XY").box(80, 60, 4, centered=(True, True, False))
result = result.faces(">Z").workplane().rect(70, 50, forConstruction=True).vertices().hole(4.5)
result = result.faces(">Z").workplane().hole(20)
'''
PLATE_REQS = [
    {"id": "valid", "type": "valid"}, {"id": "one", "type": "single_solid"},
    {"id": "size", "type": "bbox_size", "params": {"x": 80, "y": 60, "z": 4}},
    {"id": "holes", "type": "hole_count", "params": {"count": 4, "diameter": 3.4, "axis": "Z", "through": True}},
    {"id": "h1", "type": "hole_at", "params": {"center": [35, 25, 0], "diameter": 3.4, "axis": "Z"}},
    {"id": "bore", "type": "hole_at", "params": {"center": [0, 0, 0], "diameter": 20, "axis": "Z", "through": True}},
    {"id": "genus", "type": "through_hole_count", "params": {"count": 5}},
]


def test_trace_and_blame(tmp_path):
    reqs, _ = load_requirements(PLATE_REQS)
    res = Kernel(str(tmp_path)).run(PLATE_BUGGY, reqs, "p")
    assert res.success and [r.stmt for r in res.rows] == [1, 2, 3]
    traj = Trajectory(res)
    blames = localize(traj)
    assert blames[0].stmt == 2 and set(blames[0].req_ids) == {"holes", "h1"}


def test_regression_blame(tmp_path):
    code = '''import cadquery as cq
result = cq.Workplane("XY").box(40, 40, 10, centered=(True, True, False))
result = result.faces(">Z").workplane().hole(10)
result = result.faces(">Z").workplane().circle(12).cutBlind(-10)
'''
    reqs, _ = load_requirements([{"id": "bore", "type": "hole_at", "params": {"center": [0, 0, 0], "diameter": 10}}])
    traj = Trajectory(Kernel(str(tmp_path)).run(code, reqs, "r"))
    (b,) = localize(traj)
    assert b.rule == "regression" and b.stmt == 3


def test_drone_reference_defect_is_a_regression(tmp_path):
    ds = {json.loads(l)["id"]: json.loads(l) for l in open("data/dataset_v2/t3_complex_parts.jsonl", encoding="utf-8")}
    reqs, _ = load_requirements([{"id": "one", "type": "single_solid"}])
    traj = Trajectory(Kernel(str(tmp_path), timeout=120).run(ds["T3_019"]["reference_code"], reqs, "d"))
    (b,) = localize(traj)
    assert b.rule == "regression" and "separate solids" in traj.message(traj.T - 1, 0)


def test_step_rewards_telescope(tmp_path):
    reqs, _ = load_requirements(PLATE_REQS)
    traj = Trajectory(Kernel(str(tmp_path)).run(PLATE_BUGGY, reqs, "s"))
    assert abs(traj.step_rewards(1.0).sum() - traj.potential()[-1]) < 1e-9


# ---------------------------------------------------------------------------
# repair loop with a scripted model
# ---------------------------------------------------------------------------

def _fix_script(role, system, user):
    if role == "repair_local":
        assert "<<< EDIT" in user
        return json.dumps({"edits": {"S2": 'result = result.faces(">Z").workplane().rect(70, 50, '
                                           'forConstruction=True).vertices().hole(3.4)'}})
    if role == "refine_whole":
        return PLATE_BUGGY.replace("hole(4.5)", "hole(3.4)")
    raise AssertionError(role)


def test_rst_repairs_blamed_statement(tmp_path):
    reqs, _ = load_requirements(PLATE_REQS)
    eng = Engine(PLATE_PROMPT, Kernel(str(tmp_path)), ScriptedLLM(_fix_script), "plate")
    out = run_rst(eng, PLATE_BUGGY, RSTConfig(max_llm_calls=6), reqs)
    assert out.stopped == "all_pass" and out.final["all_pass"]
    assert out.usage["by_role"]["repair_local"]["calls"] == 1


def test_tests_log_baseline_runs(tmp_path):
    reqs, _ = load_requirements(PLATE_REQS)
    eng = Engine(PLATE_PROMPT, Kernel(str(tmp_path)), ScriptedLLM(_fix_script), "plate")
    out = run_tests_log(eng, PLATE_BUGGY, RSTConfig(max_llm_calls=6), reqs)
    assert out.final["all_pass"]
