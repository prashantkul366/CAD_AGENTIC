# RST: Requirement-Satisfaction Trajectories — method and code map

## Idea

A CadQuery program is a sequence of kernel states S_1 … S_T (every prefix is a real solid). Requirements
compiled from the prompt, Φ = {φ_1 … φ_N}, are evaluated on every state, giving the satisfaction matrix
M[t, i] = φ_i(S_t). Without any target geometry the matrix tells us:

- **which operation broke a requirement** (it held, then turned false: a regression),
- **which operation produced the near miss** for a requirement that was never satisfied (for a feature: the
  step that created the feature that best approximates it, e.g. the 16 mm bore where 12 mm was asked; for a size:
  the last step that changed it), or
- **that a feature was never produced** (the measurement never changed).

RST repairs only the blamed statement (plus the statements that define the bodies it uses), re-executes,
and accepts the edit only if the requirement score improves and nothing that passed now fails. The same
matrix gives potential-based step rewards r_t = Σ_i w_i[(M_t − M_{t−1})₊ − λ(M_{t−1} − M_t)₊]; with λ = 1 they
sum to Ψ(S_T) − Ψ(S_0).

Requirements have a kind: **global** (must hold whenever a solid exists, e.g. validity), **persistent** (once
true it should stay true, e.g. a hole at a position), **terminal** (only meaningful at the end, e.g. overall size).

## Code map

| File | What it does |
|---|---|
| `rst/geometry.py` | Exact kernel measurements: bbox, volume, genus (Euler–Poincaré), holes/bosses (concavity, through/blind, partial arcs), planar faces, point classification, symmetry by point-to-mesh distance |
| `rst/requirements.py` | The requirement DSL: 22 typed predicates with tolerances, kinds, and a "measure" used by the footprint rule; the catalogue text shown to the LLM writer |
| `rst/tracer.py`, `rst/runner.py`, `rst/kernel.py` | One execution records every intermediate solid (hook on `Workplane.newObject`), maps it to its statement, and evaluates every requirement on every state; isolated process without credentials; persistent worker pool |
| `rst/program.py` | Statement-level view of a program: line→statement map, def-use dependencies, splicing edits |
| `rst/matrix.py` | Establishment, regressions, potential and step rewards |
| `rst/localize.py` | Blame rules (regression / last-touch / missing) and baseline localisers (LLM, random, last statement) |
| `rst/roles.py` | Shared LLM roles: generator, requirement writer, local repair, whole-program refine, ReAct, error fixer |
| `rst/pipeline.py` | RST and the baselines on one equal-budget engine (shared P0, shared requirements) |
| `rst/cadsmith_fixed.py` | CADSmith with its verified bugs fixed (judge failure = fail, no-vision prompt, keep last valid program; optional leak-free retrieval) |
| `rst/mutants.py`, `rst/autospec.py`, `rst/thread_program.py` | Fault injection (E3), oracle requirement suites from a reference solid, conversion to one-feature-per-statement form (E1) |
| `rst/evaluate.py`, `rst/cadtests_runner.py` | Held-out evaluators: CADTestBench tests (PR/RS/IR), requirement suites, shape metrics (CADSmith protocol, corrected protocol, Text2CAD protocol) |
| `rst/datasets.py` | Loaders for cadsmith, hardlong, cadtestbench-{detailed,abstract}, text2cad |

## Experiments

| Script | Experiment | Needs |
|---|---|---|
| `scripts/rst/e1_trace_coverage.py` | E1: tracing coverage and conversion | CPU |
| `scripts/rst/e3_localization.py` | E3: blame accuracy on injected faults (oracle / entry / self-written specs; matrix vs LLM vs random) | CPU (+ Bedrock for `--spec self`, `--llm`) |
| `scripts/rst/e4_regressions.py` | E4: do whole-program rewrites break satisfied requirements? | CPU |
| `scripts/rst/e0_rescore.py` | E0: re-score the CADSmith reproduction under the corrected protocol | CPU |
| `scripts/rst/run_benchmark.py` | main runs (all methods and ablations) | Bedrock |
| `scripts/rst/evaluate_run.py` | held-out scoring of a run, paired tests, complexity bins | CPU |
| `scripts/rst/validate_cadtests_eval.py` | checks our CADTestBench evaluator against the paper's Table 3 | CPU |

## Blame-rule development protocol (to report in the paper)

The blame rules were developed by inspecting E3 misses on **mutant seed 0** (Hard-Long and CADTestBench
references). Changes made during development: near-miss provenance; plausibility limits on near misses;
feature-level requirements ranked before whole-part totals; features untouched since the base body count as
missing; alternatives for regressions at whole-part moves; transitive parameter regions; value provenance
for features built on the wrong plane (size + position literals); and (v2) for non-monotone constructions,
when a lost requirement's feature is touched again later, that later step is blamed first. **The rules are
frozen at v2**, developed only on seed 0; reported E3 numbers use **mutant seeds 1 and 2**, which no rule was
tuned on.

**v3** (frozen 2026-10-01, before any held-out number was computed with it; developed on seed 0 only):
- *unexplained operations*: when a feature never appears and no statement's numbers match it (or only the
  base body ever touched it), blame the first step that changed no requirement's measurement or verdict,
  e.g. a cut whose tool is on the wrong plane or misses the part;
- *whole tool-body chains*: a tool body built over several statements (`ribs = ribs.union(...)`) is in the
  repair region in full (`Program.dependencies(chains=True)`).

Seed-0 effect, first guess (every localiser scored with the same region): Hard-Long 55.9 → 60.2 %,
CADTestBench 54.3 → 57.2 %. v2 stays the method's default (`RULES_VERSION`) until v3 is confirmed on held-out
data: mutant seeds 1–2 of Hard-Long v0 and CADTestBench, and the 40 Hard-Long v1 parts, which were written after
v2 was frozen and never inspected during rule development. `rst_v3` in `run_benchmark.py` runs the method with v3.

**Re-scoring without re-execution.** Every E3 fault keeps its program, requirements and full trajectory under
`runs/e3/<tag>/work/`. `scripts/rst/e3_rescore.py --rules <v> [--region <v>]` rebuilds the trajectories and
applies any rule version and every baseline (random replays its original stream) in about a minute; with
`--check` it reproduces the stored records exactly (0 differences on all held-out runs).

**Spectrum-based fault localisation baselines** (`sbfl_rank`): Ochiai, Tarantula and DStar with requirements
as tests; a statement covers a requirement when it changed that requirement's measurement or verdict.
On the development seed, SBFL is as good as the matrix on short CADTestBench programs (Ochiai 56.9 % vs v2 54.3 %,
v3 57.2 %) but far behind on Hard-Long (39 % vs 56–60 %): the blame rules pay off on long programs.

## Confirmatory test on Hard-Long v1 (plan fixed before running)

Data: the 40 Hard-Long v1 parts (HL_031–HL_070), written after v2 was frozen and never inspected during rule
development, so all mutant seeds (0, 1, 2) are held-out; 8 mutants per part and seed, own suites (`--spec entry`).
Run once with `e3_localization.py --dataset hardlong-v1`, then re-score the same faults with `e3_rescore.py`.

Pre-stated comparisons (pooled over seeds; 95 % Wilson intervals; paired exact McNemar tests on the same faults):
1. matrix v3 vs matrix v2, first guess (expected: v3 ≥ v2);
2. matrix v3 vs SBFL DStar and Ochiai, first guess (expected: matrix higher on Hard-Long);
3. matrix v3 vs random and last step (expected: well above both);
4. per fault type, descriptive only.
Whatever the outcome, v1 numbers are reported as they come out; the version adopted as the method default
is v3 if comparison 1 is not worse, otherwise v2. No rule is changed after this run.

## Not yet implemented

Learning components (kernel-labelled step scorer / PRM, process-vs-outcome GRPO) — deliberately after the
week-5 go/no-go. E2 (DSL fidelity of self-written requirements against CADTests; needs Bedrock).
