# Numbers to beat

All values are copied from each paper's own tables (arXiv HTML, parsed Sept 2026) or measured in this repo.
Protocols differ between papers, so compare only within a benchmark and re-run baselines under one protocol.

## 1. CADTestBench (main benchmark) — arXiv 2605.07807, Table 3

200 CADPrompt objects (CADPrompt is from CADCodeVerify, ICLR 2025), each with a *detailed* and an *abstract*
prompt; 5,937 executable tests (~15 per sample) grouped into requirements.
**PR** = share of samples passing all tests; **RS** = share of requirement groups satisfied; **IR** = share of
programs that fail to run (counted as failures everywhere).

| Method | LLM | Detailed IR↓ | Detailed RS↑ | Detailed PR↑ | Abstract IR↓ | Abstract RS↑ | Abstract PR↑ |
|---|---|---|---|---|---|---|---|
| Text2CAD (fine-tuned) | – | 0.005 | 0.405 | 0.025 | 0.015 | 0.603 | 0.085 |
| CADCodeVerify | GPT-4(.1) | 0.025 | 0.794 | 0.410 | 0.040 | 0.880 | 0.630 |
| 10-shot | Claude-4.6-Sonnet | 0.155 | 0.744 | 0.510 | 0.115 | 0.830 | 0.640 |
| ReAct | Claude-4.6-Sonnet | 0 | 0.874 | 0.580 | 0 | 0.929 | 0.715 |
| ReAct | GPT-5.2 | 0 | 0.835 | 0.480 | 0.005 | 0.916 | 0.695 |
| CADTests | Claude-4.6-Sonnet | 0.005 | 0.882 | 0.590 | 0 | 0.953 | 0.765 |
| **CADTests + Log** | **Claude-4.6-Sonnet** | **0** | **0.897** | **0.625** | **0** | **0.962** | **0.810** |

**Targets for RST (same LLM, Claude Sonnet 4.6, equal LLM-call budget):**
Detailed PR ≥ 0.675 (+5 pts), Abstract PR ≥ 0.86 (+5 pts); RS and IR no worse.
Human agreement (Table 4): CADTests RS AUC 0.928 vs Chamfer 0.663, CLIP 0.665, LVM judge 0.659.

**Our evaluator vs the paper** (`scripts/rst/validate_cadtests_eval.py`):
- The 200 reference programs pass 100 % of their own tests on both splits (PR = RS = 1.0, IR = 0).
- The released `baselines/` folders for Claude-4.6-Sonnet (`CADTests`, `CADTests_Log`, top-level) are
  **byte-identical** (400/400 files); their embedded export paths name the run `claude_react_cadtests_final_*`,
  i.e. the **CADTests** row. Re-scored here: detailed PR 0.56 / RS 0.854 / IR 0.04 (paper 0.59 / 0.882 / 0.005),
  abstract PR 0.755 / RS 0.932 / IR 0.025 (paper 0.765 / 0.953 / 0).
- The whole gap is the invalid ratio: 8 detailed and 5 abstract released programs fail **under CadQuery 2.6.1**
  (their own embedded self-checks fail on different geometry, one OCP constructor changed). CADTestBench only
  requires `cadquery>=2.4`, so published numbers are environment-dependent.
- **Like-for-like reference for our tables:** the released CADTests (Claude-4.6-Sonnet) outputs re-scored in our
  environment — detailed PR 0.56, abstract PR 0.755 — alongside the paper's reported numbers.

## 2. Text2CAD test set (8,046 DeepCAD models; standard for trained text-to-CAD models)

Chamfer distance ×10³ on normalised shapes (Text2CAD protocol: 8,192 points, each cloud divided by its
coordinate range, no alignment, invalid outputs excluded); IR in %.

| Method | Venue | Median CD↓ | Mean CD↓ | IR %↓ | Note |
|---|---|---|---|---|---|
| Text2CAD | NeurIPS'24 | 0.37 | 26.41 | 0.93 (3.5–3.75 when re-run by others) | L3 prompts |
| CADFusion | ICML'25 | – | 19.89 | 6.20 | own split and prompts |
| Text-to-CadQuery (Qwen2.5-3B) | arXiv | 0.191 | 10.23 | 6.5 | own 90/5/5 split |
| **CAD-Coder** (SFT+CoT+GRPO) | NeurIPS'25 | **0.17** | 6.54 | 1.45 | "official" test set, possibly filtered |
| cadrille (text) | desk-rejected ICLR'26 | 0.20 | **3.95** | 1.4 | IoU 82.1 % |

We evaluate on a fixed 500-prompt subset of the CAD-Coder release of this split (`data/text2cad_subset_ids.txt`)
with the exact Text2CAD protocol (`rst/evaluate.py::text2cad_cd`) plus millimetre metrics.
**Target:** RST on top of an open 7B coder ≈ the best trained model; RST improves every generator it wraps.
These prompts are short sketch-and-extrude parts, so the expected gain here is small (comparability, not the headline).

## 3. CADCodeVerify on CADPrompt (ICLR'25, Table 2; median (IQR), GPT-4 few-shot)

| Feedback | IoGT↑ | PC distance↓ | Hausdorff↓ | Compile % |
|---|---|---|---|---|
| none (generated) | 0.939 | 0.155 | 0.494 | 96.0 |
| CADCodeVerify | 0.944 | 0.127 | 0.419 | 96.5 |
| geometric solver (uses ground truth) | 0.944 | 0.103 | 0.399 | 95.5 |

## 4. CADSmith (arXiv 2603.26512) — paper vs our reproduction (Bedrock, Sonnet 4.5 + Opus 4.5 judge)

| Config | Exec % | CD med | CD mean | F1 med | IoU med |
|---|---|---|---|---|---|
| Zero-shot (paper / ours) | 95 / 92 | 0.55 / 0.59 | 28.37 / 21.51 | 0.9707 / 0.9648 | 0.8085 / 0.8848 |
| No vision (paper / ours) | 99 / 100 | 0.48 / 0.44 | 18.19 / 3.43 | 0.9792 / 0.9877 | 0.9563 / 0.9555 |
| Full vision (paper / ours) | 100 / 100 | 0.48 / 0.44 | 0.74 / 0.66 | 0.9846 / 0.9869 | 0.9629 / 0.9720 |

The vision effect does not reproduce (T3 mean CD 1.26 vs 1.22 without/with vision; paper 49.68 vs 1.42), IoU varies
by up to 0.68 on identical geometry, and the no-vision judge "sees" renders in 83/116 verdicts.

**E0: corrected re-scoring of the reproduction** (`scripts/rst/e0_rescore.py`; parts rebuilt from the stored code;
rigid ICP only, exact OCCT boolean IoU, 3 sampling seeds):

| Run | mean CD (CADSmith protocol → corrected) | median CD | CD noise floor | exact IoU as placed (median) | bbox error > 1 mm | volume error > 5 % |
|---|---|---|---|---|---|---|
| Full vision | 0.66 → 0.63 | 0.44 | 0.437 | 1.00 | 2 / 100 | 2 / 100 |
| No vision | 3.52 → **1.44** | 0.45 | 0.437 | 1.00 | 3 / 100 | 4 / 100 |
| Zero-shot | 20.4 → 22.1 | 0.59 | 0.470 | 1.00 | 12 / 100 | 11 / 100 |

Median CD is the sampling noise floor (a part compared with itself), so it cannot rank methods. Part of the
no-vision gap under the original protocol was an alignment artefact (scale-fitting ICP on T2_020). The ordering
full < no-vision < zero-shot survives; dimensional errors (bbox, volume) make the differences visible.

## 5. Premise experiments measured so far (this repo, CPU only)

| Experiment | Result | Gate |
|---|---|---|
| E1 CADSmith references: trace + convert to one-feature-per-statement | 100 % execute, 100 % convert with identical solids; mean 2.3 states, 1.6 → 2.2 modifying statements; only 11 % → 29 % have ≥ 3 steps | G0 (≥ 70 % convert) **passed**; CADSmith-100 too short for step-level attribution |
| E4 CADSmith reproduction, whole-program refinement rounds (held-out = oracle requirements from the reference) | full vision: 3/20 rounds (15 %) broke a satisfied requirement; no vision: 5/16 (31 %) | E4 (≥ 15 %) met on both, small n |
| **E3 attribution, CADTestBench references** (oracle spec, threaded; held-out seeds 1–2, rules v2; 588 faults) | matrix first blame **57 %** (95 % CI 53–61), top-3 **84 %**; random 45 %, last step 39 % | first-blame gate (≥ 70 %) not met; short programs make random strong |
| **E3 attribution, Hard-Long** (own suites; held-out seeds 1–2, rules v2; 380 faults, ~9.5 steps) | matrix first blame **57 %** (52–62), top-3 **68 %**; random **11 %**, last step **16 %**; deleted features 70 %, moved features 71 %, wrong counts 55 %, parameter changes 45 %, wrong plane 29 % | ~5x random; first-blame gate not met; LLM comparison pending (Bedrock) |
| E3 development set (seed 0; rules were tuned on it) | CADTestBench 55 % / 80 %; Hard-Long 56 % / 67 % | not reported as a result |

## Final table we are aiming for (paper Table 1)

Rows grouped by base generator (Claude Sonnet 4.6; an open 7B coder; trained CadQuery models + RST);
columns: CADTestBench detailed PR/RS, abstract PR/RS, Hard-Long PR/RS, invalid %, LLM calls, kernel calls.
Methods: zero-shot, best-of-N with self-test vote, ReAct, CADSmith (bug-fixed), CADTests+Log (published and our
re-implementation), **RST**, and the oracle-specification upper bound.
