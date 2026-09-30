# Hard-Long (seed v0): long-horizon text-to-CAD prompts

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
| `check_entry.py` | verify one entry (and append it to a file): suite parses, reference passes every requirement, ≥ 8 traced states, 3 single-statement mutants all caught |
| `verify.py` | re-verify all entries with the current code |
| `merge_parts.py` | merge authoring batches from `parts/` |

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
