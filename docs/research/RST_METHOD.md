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

## Not yet implemented

Learning components (kernel-labelled step scorer / PRM, process-vs-outcome GRPO) — deliberately after the
week-5 go/no-go; CADCodeVerify re-implementation; open-weight baselines (CAD-Coder, ProCAD) inference on a GPU.
