# Hard-Long: long-horizon text-to-CAD prompts (v0: 30 parts, v1: 40 parts)

30 realistic mechanical parts, each described in an engineer's words (160–220 words, full dimensions and placement,
no CadQuery vocabulary), with a hand-written CadQuery reference that needs **at least 8 construction steps** with
interacting features (fillets after holes, shells with bosses, patterns on offset faces, counterbores, keyways,
cross-drilled ports), and a **held-out requirement suite** in the RST requirement language (603 requirements in
total, 18–25 per part).

Existing benchmarks are dominated by short programs (CADSmith-100: 2.3 construction states on average, only 11 %
with 3 or more steps), where step-level attribution has little to work with. Hard-Long is where RST's claim should
show: the gain should grow with the number of operations.

**Status: machine-authored, must be audited by a human engineer before it is used in a paper.** Every entry is
machine-verified (below), but the prompts' engineering intent, the reference geometry and the requirement choices
need a human pass.

## Files

| File | Content |
|---|---|
| `hard_long_v0.jsonl` | one entry per line: id, family, prompt, reference_code, requirements, n_ops, n_rows, volume, bbox, mutant_kills, notes |
| `hard_long_v1.jsonl` | 40 further parts (HL_031–HL_070), same format; held out from blame-rule development (below) |
| `check_entry.py` | verify one entry (and append it to a file): suite parses, reference passes every requirement, ≥ 8 traced states, 3 single-statement mutants all caught |
| `verify.py` | re-verify all entries with the current code (`--version v1` for v1) |
| `merge_parts.py` | merge authoring batches from `parts/` (v0) or `parts_v1/` (`--version v1`) |

## Verification rules

1. The reference executes to a valid single solid and passes every requirement of its suite.
2. At least 8 construction states are traced on the reference.
3. Three geometry-changing single-statement mutants of the reference each fail at least one requirement.
4. Requirements check only what the prompt states, each with the prompt phrase it checks in `text`; tolerances were
   never loosened to make a reference pass (weak suites were strengthened with extra probes instead).

## Notes for the human audit (from authoring)

- HL_003: called "weld-neck style" but the hub is a straight cylinder (hubbed / slip-on may be the better name).
- HL_004, HL_017, HL_018: keyed bores are not complete cylinders, so they are checked with material probes and the keyway floor plane.
- HL_007: the port connectivity (P → A and gauge, T → B) is an authoring choice; the through-opening count depends on it.
- HL_008 / HL_009: through-opening counts (3 and 6) follow from the described cavity and knuckle layout.
- HL_018, HL_021: tooth counts are checked by counting tip / root faces (keyway and holes break rotational symmetry).
- HL_019: the rim has a rectangular section with R4 fillets (a torus rim gave invalid unions in this OCCT build).
- HL_025: one gusset only, as described, so no symmetry check.
- HL_026: exactly 8 construction states (the minimum).
- HL_027: several probes sit 0.2–0.3 mm from groove surfaces (correct, but close to the boundary).
- HL_029: "folded flanges" interpreted as a hat section with outward feet.
- Coaxial features (counterbore + drill, bore inside a hub) were originally checked with `coaxial` + probes because
  `hole_at` picked the first coaxial cylinder; `hole_at` now prefers the one with the right diameter.

## Parts

30 parts, 603 requirements, 8–12 construction states (mean 9.5).

| id | family | construction states | requirements | mutants caught | bbox (mm) |
|---|---|---|---|---|---|
| HL_001 | bracket | 10 | 20 | 3/3 | 80 × 60 × 70 |
| HL_002 | bracket | 11 | 25 | 3/3 | 100 × 40 × 65 |
| HL_003 | pipe flange | 9 | 20 | 3/3 | 160 × 160 × 55 |
| HL_004 | shaft coupling | 9 | 20 | 3/3 | 100 × 100 × 50 |
| HL_005 | pillow block | 10 | 20 | 3/3 | 160 × 50 × 96 |
| HL_006 | flanged bearing housing | 10 | 20 | 3/3 | 100 × 100 × 46 |
| HL_007 | hydraulic manifold | 10 | 20 | 3/3 | 90 × 60 × 50 |
| HL_008 | valve body | 10 | 20 | 3/3 | 100 × 60 × 60 |
| HL_009 | hinge leaf | 9 | 20 | 3/3 | 100 × 45 × 10 |
| HL_010 | enclosure | 8 | 20 | 3/3 | 120 × 80 × 43 |
| HL_011 | gearbox housing | 12 | 20 | 3/3 | 210 × 130 × 52 |
| HL_012 | junction box | 11 | 20 | 3/3 | 150 × 80 × 50 |
| HL_013 | heat sink | 8 | 20 | 3/3 | 80 × 100 × 38 |
| HL_014 | pin fin heat sink | 9 | 20 | 3/3 | 60 × 60 × 25 |
| HL_015 | control knob | 9 | 20 | 3/3 | 42 × 42 × 18 |
| HL_016 | pull handle | 9 | 20 | 3/3 | 124 × 24 × 38 |
| HL_017 | pulley | 10 | 20 | 3/3 | 120 × 120 × 40 |
| HL_018 | gear | 9 | 20 | 3/3 | 100 × 100 × 30 |
| HL_019 | handwheel | 10 | 20 | 3/3 | 200 × 200 × 38 |
| HL_020 | shaft | 10 | 20 | 3/3 | 190 × 40 × 40 |
| HL_021 | shaft | 9 | 20 | 3/3 | 36 × 36 × 90 |
| HL_022 | axle | 9 | 18 | 3/3 | 90 × 90 × 132 |
| HL_023 | clamp | 10 | 20 | 3/3 | 90 × 40 × 60 |
| HL_024 | motor plate | 9 | 20 | 3/3 | 90 × 80 × 40 |
| HL_025 | motor bracket | 10 | 20 | 3/3 | 70 × 60 × 80 |
| HL_026 | fixture plate | 8 | 20 | 3/3 | 160 × 120 × 20 |
| HL_027 | vise jaw | 9 | 20 | 3/3 | 150 × 16 × 45 |
| HL_028 | sheet metal bracket | 10 | 20 | 3/3 | 62 × 40 × 40 |
| HL_029 | sheet metal cover | 9 | 20 | 3/3 | 120 × 104 × 20 |
| HL_030 | pipe clamp | 9 | 20 | 3/3 | 144 × 30 × 46 |

## v1: 40 parts held out from rule development

Written after blame rules v2 were frozen (2026-09-30/10-01) by four authoring batches, verified with the same rules
(all 40 pass; three entries were strengthened after an extended sweep of up to ~36 mutants each, HL_031–HL_035).
No part of v1 was looked at while developing any blame rule, so every mutant seed of v1 is held-out data for E3
(see `docs/research/RST_METHOD.md`, "Confirmatory test on Hard-Long v1"). 40 parts,
879 requirements, 8–13 construction states,
8–13 solid-changing steps (mean 9.6).

Notes for the human audit (from authoring; each entry's `notes` has the details):
- Check values derived by reasoning or read from the kernel rather than stated in the prompt: HL_037 (cone-face
  count 14), HL_039 (through-opening count 4), HL_051 (bore depth 32), HL_052 (hub fillet split into 4 faces),
  HL_057 (corner fillets 7 faces), HL_059 (9 through-openings), HL_060 (boss heights 11.5 / 6.5).
- Interpretations: HL_042 slot chambers are rectangles; HL_062 "stiffening flange" is a rear web; HL_063 spline is
  6 × 26 × 32 (confirm the series); HL_068 "bead-like rib" is a rectangular embossed rib, "bent flange" an upturned
  lip; HL_032 tooth form is simplified (round seat plus parallel flanks).
- Weaker checks: HL_045 blends counted as B-spline faces; HL_054 countersinks and HL_059 outer chamfer by cone-face
  count plus one probe; HL_048 ring-groove depth only indirectly; HL_031 junction fillet by probes.
- Checker limitation: cylinders sharing an axis and radius merge into one feature, so HL_062 spot faces, HL_065
  retaining-ring grooves and HL_031 aligned flange bolt holes are checked with material probes.
- Close probes: HL_069 groove-wall probes 0.2 mm from the wall.
- Family also in v0: HL_068 (sheet metal bracket), a different design.

| id | family | construction states | requirements | mutants caught | bbox (mm) |
|---|---|---|---|---|---|
| HL_031 | pipe fitting | 13 | 22 | 3/3 | 160 × 135 × 120 |
| HL_032 | sprocket | 9 | 22 | 3/3 | 75.8 × 76 × 25 |
| HL_033 | cam | 9 | 21 | 3/3 | 90 × 90 × 28 |
| HL_034 | lever | 10 | 22 | 3/3 | 152 × 40 × 20 |
| HL_035 | connecting rod | 10 | 22 | 3/3 | 183 × 56 × 22 |
| HL_036 | end cap | 9 | 22 | 3/3 | 110 × 110 × 18 |
| HL_037 | standoff | 11 | 22 | 3/3 | 19.6 × 17 × 40 |
| HL_038 | sheet bracket | 8 | 22 | 3/3 | 50 × 75 × 40 |
| HL_039 | fork end | 10 | 22 | 3/3 | 112 × 44 × 42 |
| HL_040 | quick release plate | 8 | 22 | 3/3 | 50 × 38 × 10 |
| HL_041 | impeller | 9 | 22 | 3/3 | 140 × 140 × 34 |
| HL_042 | T-slot extrusion | 10 | 22 | 3/3 | 40 × 40 × 100 |
| HL_043 | shelf bracket | 9 | 22 | 3/3 | 40 × 150 × 130 |
| HL_044 | enclosure lid | 9 | 22 | 3/3 | 120 × 80 × 14 |
| HL_045 | tensioner arm | 9 | 22 | 3/3 | 155 × 52 × 24 |
| HL_046 | rocker arm | 9 | 22 | 3/3 | 117 × 24 × 30 |
| HL_047 | adapter flange | 9 | 22 | 3/3 | 160 × 160 × 22 |
| HL_048 | heat-exchanger end plate | 9 | 22 | 3/3 | 200 × 200 × 20 |
| HL_049 | PCB tray | 10 | 22 | 3/3 | 134 × 80 × 20 |
| HL_050 | corner bracket | 10 | 22 | 3/3 | 50 × 50 × 50 |
| HL_051 | hydraulic end cap | 9 | 22 | 3/3 | 90 × 100 × 42 |
| HL_052 | gearmotor flange | 10 | 22 | 3/3 | 110 × 110 × 44 |
| HL_053 | cable clamp saddle | 8 | 22 | 3/3 | 100 × 30 × 16 |
| HL_054 | latch body | 9 | 22 | 3/3 | 80 × 30 × 28 |
| HL_055 | seat clamp | 12 | 22 | 3/3 | 40 × 50 × 14 |
| HL_056 | drone motor mount | 9 | 22 | 3/3 | 90 × 60 × 6 |
| HL_057 | tool post block | 9 | 22 | 3/3 | 60 × 60 × 50 |
| HL_058 | pipe bracket | 11 | 22 | 3/3 | 60 × 130 × 96 |
| HL_059 | fan guard ring | 10 | 22 | 3/3 | 124 × 124 × 12 |
| HL_060 | insert boss plate | 9 | 22 | 3/3 | 80 × 60 × 14 |
| HL_061 | caster top plate | 9 | 22 | 3/3 | 100 × 80 × 18 |
| HL_062 | conveyor idler bracket | 10 | 22 | 3/3 | 140 × 50 × 85 |
| HL_063 | splined hub | 10 | 22 | 3/3 | 100 × 100 × 40 |
| HL_064 | gear pump body | 10 | 22 | 3/3 | 110 × 70 × 30 |
| HL_065 | linear bearing pillow block | 10 | 22 | 3/3 | 50 × 80 × 40 |
| HL_066 | nozzle adapter | 9 | 22 | 3/3 | 40 × 36 × 60 |
| HL_067 | terminal block | 9 | 22 | 3/3 | 80 × 30 × 22 |
| HL_068 | sheet metal bracket | 11 | 22 | 3/3 | 70 × 50 × 50 |
| HL_069 | machine foot | 10 | 22 | 3/3 | 80 × 80 × 30 |
| HL_070 | manifold cover plate | 9 | 22 | 3/3 | 120 × 80 × 22 |
