"""Tests for the experiment tools: mutants, oracle specs, program threading, CADTestBench runner, metrics."""

import json

from rst.evaluate import run_cadtests
from rst.kernel import Kernel
from rst.localize import localize
from rst.matrix import Trajectory
from rst.mutants import make_mutants, normalised
from rst.requirements import load_requirements
from rst.thread_program import thread_program

CHAIN = ('import cadquery as cq\n'
         'result = cq.Workplane("XY").box(40, 30, 6).faces(">Z").workplane().rect(30, 20, forConstruction=True)'
         '.vertices().hole(4).edges("|Z").fillet(3)\n')

THREADED = '''import cadquery as cq
result = cq.Workplane("XY").box(60, 40, 8, centered=(True, True, False))
result = result.faces(">Z").workplane().rect(50, 30, forConstruction=True).vertices().hole(5)
result = result.faces(">Z").workplane().hole(12)
boss = cq.Workplane("XY").workplane(offset=8).circle(10).extrude(6)
result = result.union(boss)
result = result.faces(">Z").workplane().hole(6)
result = result.edges("|Z").fillet(2)
'''


def test_thread_program_equivalent(tmp_path):
    thr, added = thread_program(CHAIN)
    assert added == 2 and thr.count("result = result.") == 2
    k = Kernel(str(tmp_path))
    a, b = k.run(CHAIN, [], "a", export=False), k.run(thr, [], "b", export=False)
    assert a.success and b.success
    assert abs(a.geometry["volume"] - b.geometry["volume"]) < 1e-6
    assert len({r.stmt for r in b.rows}) == 3


def test_mutants_change_one_statement():
    ms = make_mutants(THREADED, 8, seed=1, extent=60)
    assert len(ms) >= 5
    base = normalised(THREADED).split("\n")
    for m in ms:
        diff = [i for i, (x, y) in enumerate(zip(base, m.code.split("\n"))) if x != y]
        assert len(diff) == 1, m.detail
    assert {m.kind for m in ms} >= {"param_shift", "feature_delete"}


def test_autospec_passes_reference_and_localises_fault(tmp_path):
    k = Kernel(str(tmp_path))
    ref = k.run(THREADED, [], "ref", export=False, autospec=True)
    reqs, errors = load_requirements(ref.autospec)
    assert not errors and len(reqs) >= 8
    assert k.run(THREADED, reqs, "ref2", export=False).all_pass()
    bad = THREADED.replace("hole(12)", "hole(16)")
    traj = Trajectory(k.run(bad, reqs, "bad", export=False))
    assert localize(traj)[0].stmt == 3


def test_cadtests_runner_protocol(tmp_path):
    tests = [
        {"cadtest_id": 1, "requirement_id": "solid", "cadtest_code":
            "n = final_result.solids().size()\ncheck(n == 1, 'ok', f'found {n}')"},
        {"cadtest_id": 2, "requirement_id": "size", "cadtest_code":
            "bb = final_result.val().BoundingBox()\ncheck(abs(bb.xlen - 60) < 0.1, 'ok', 'x')"},
        {"cadtest_id": 3, "requirement_id": "size", "cadtest_code":
            "bb = final_result.val().BoundingBox()\ncheck(abs(bb.zlen - 99) < 0.1, 'ok', 'z')"},
    ]
    r = run_cadtests(THREADED, tests, str(tmp_path), "ct")
    assert not r["invalid"] and not r["pass"] and r["rs"] == 0.5 and r["tests_passed"] == 2
    exported = "import cadquery as cq\npart = cq.Workplane().box(60, 1, 1)\ncq.exporters.export(part, 'x.stl')\n"
    r2 = run_cadtests(exported, tests[:2], str(tmp_path), "ct2")
    assert r2["pass"]
    r3 = run_cadtests("raise ValueError('x')", tests, str(tmp_path), "ct3")
    assert r3["invalid"] and r3["rs"] == 0.0
