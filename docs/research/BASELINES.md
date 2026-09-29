# Baselines: what can be run, and how

Venues verified on OpenReview / official proceedings (Sept 2026). GPU needs are estimates from weight sizes.

| Method | Venue | Output | How we run it | Needs |
|---|---|---|---|---|
| Zero-shot (shared P0) | – | CadQuery | `run_benchmark.py --methods zero_shot` | Bedrock |
| ReAct (execution + measurements feedback) | CADTests baseline | CadQuery | `react` | Bedrock |
| CADTests / CADTests+Log (published outputs) | arXiv (under review) | CadQuery | re-scored from their released programs (`validate_cadtests_eval.py`) | CPU |
| CADTests+Log-style (our re-implementation on our DSL) | – | CadQuery | `tests_log` | Bedrock |
| Best-of-N with self-test vote (CodeT-style) | – | CadQuery | `best_of_n` | Bedrock |
| CADSmith, bugs fixed (+ no-leak, no-vision variants) | arXiv | CadQuery | `cadsmith_fixed`, `cadsmith_fixed_noleak`, `cadsmith_fixed_novision` | Bedrock (Opus judge) |
| CADCodeVerify | ICLR'25 | CadQuery | re-implementation from the paper's prompts (App. B) — not yet written | Bedrock |
| CAD-Coder (`gudo7208/CAD-Coder`, Qwen2.5-7B) | NeurIPS'25 | CadQuery | inference script — not yet written | 1×A100 40GB (Colab) |
| ProCAD coder (`BBexist/ProCAD-coder`, Qwen2.5-7B) | ICML'26 | CadQuery | same | 1×A100 40GB |
| Text-to-CadQuery (`ricemonster/qwen2.5-3B-SFT`) | arXiv | CadQuery | same | small GPU |
| cadrille (`maksimko123/cadrille`, 2B) | desk-rejected ICLR'26 (cite carefully) | CadQuery | repo `test.py --mode text` | ≥16 GB GPU |
| Text2CAD, CADFusion | NeurIPS'24, ICML'25 | sketch-extrude sequences | need sequence → B-Rep conversion | small GPU |
| FutureCAD, ToolCAD, RA-CAD, PR-CAD | various | – | no code/weights: cite only | – |

RST ablations (all `run_benchmark.py` presets): `rst_llm_localize`, `rst_random_localize`, `rst_last_localize`
(localisation source), `rst_whole_rewrite` (repair scope), `rst_accept_always`, `rst_accept_improve` (acceptance rule),
plus the oracle-specification condition in E3.
