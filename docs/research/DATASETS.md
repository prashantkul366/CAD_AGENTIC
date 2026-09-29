# Datasets: sources, licences, and what is committed

This repository is public, so third-party data is **fetched, not committed**. Everything third-party lives in
`data/external/` (git-ignored) and is rebuilt identically by the fetch scripts below.

| Dataset | Role | Source | Licence | In repo? | How to get it |
|---|---|---|---|---|---|
| CADSmith dataset_v2 (100 prompts, T1–T3) | continuity with the paper we build on | jabarkle/CADSmith (already in this fork) | none stated upstream | yes (was already in the fork) | — |
| **Hard-Long** (ours, 30 parts, ≥ 8 operations) | where step-level attribution should matter | written for this project | ours | **yes**: `data/hard_long/` | — |
| **CADTestBench** tests + prompts (200 × detailed/abstract, 5,937 tests) | main benchmark | HF `dimitrismallis/CADTestBench` | MIT (tests); prompts are verbatim CADPrompt | no | `python scripts/data/fetch_cadtestbench.py` |
| CADPrompt reference programs (200) | references for CADTestBench | github.com/Kamel773/CAD_Code_Generation (CADCodeVerify, ICLR'25) | **no licence file** (all rights reserved by default) | no | fetched by the same script |
| CADTestBench released outputs (Claude-4.6-Sonnet, GPT-5.2) | evaluator validation, E4 | github.com/dimitrismallis/CADTestBench `baselines/` | MIT | no | `python scripts/data/fetch_cadtestbench_baselines.py` |
| **Text2CAD test subset** (500 of 8,046, L3 prompts + CadQuery references) | comparability with trained models | HF `gudo7208/CAD-Coder` (`cad_data_test_cot.json`) | card says Apache-2.0, but derived from Text2CAD (CC BY-NC-SA 4.0): treat as **CC BY-NC-SA 4.0, non-commercial** | only the id list: `data/text2cad_subset_ids.txt` | `python scripts/data/fetch_text2cad.py --ids-file data/text2cad_subset_ids.txt` |

Notes
- CADPrompt/CADTestBench contamination: only 13 of the 200 CADPrompt shapes are in the Text2CAD/DeepCAD test
  split; about 132 are in cadrille's training split. Models fine-tuned on Text2CAD may have seen most CADTestBench shapes.
- The original Text2CAD dataset (HF `SadilKhan/Text2CAD`) is gated (login + click-through); it is only needed for
  the other prompt levels (abstract / beginner / intermediate), e.g. for the clarification follow-up.
- Alternative Text2CAD+CadQuery release: HF `maksimko123/text2cad` (cadrille, CC BY-NC 4.0, 7,797 test programs).
- `data/dataset_v2/reference_stls/` in the fork is stale (e.g. T1_031 does not match its reference code; T1_046–050
  missing). The benchmarks regenerate references at run time; do not use those STLs.
