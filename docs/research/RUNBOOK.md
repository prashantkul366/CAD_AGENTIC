# Runbook: the Bedrock stages (run on the machine with AWS access)

Everything that calls Claude is in these steps. Everything else (evaluation, experiments without an LLM,
data preparation) runs on any machine. After each stage, zip the run and send it back; the analysis is done
from the zip.

## 0. One-time setup

```bash
git fetch origin
git checkout research_idea
git pull
pip install -r requirements.txt          # adds rtree and pyarrow to the reproduction environment
python -m pytest tests -q                 # CPU only, ~2 min, should end with "40 passed"
```

`.env` is the same as for the CADSmith reproduction (`LLM_BACKEND=bedrock`, AWS credentials, `AWS_REGION`).

Fetch the benchmarks (not committed for licence reasons, see `DATASETS.md`):

```bash
python scripts/data/fetch_cadtestbench.py
python scripts/data/fetch_text2cad.py --ids-file data/text2cad_subset_ids.txt
```

Check model access, including the model the RST runs use (Claude Sonnet 4.6, to match CADTestBench):

```bash
python scripts/check_llm.py --model claude-sonnet-4-6
```

Every line must end with `'OK'`. If the `extra:claude-sonnet-4-6` line fails, copy the exact Sonnet 4.6
model / inference-profile ID from the Bedrock console and pass it as `--model <that id>` in every command below.

## 1. Smoke test (a few minutes, well under $1)

```bash
python scripts/rst/run_benchmark.py --exp smoke --datasets hardlong --limit 2 --methods zero_shot,react,tests_log,rst
python scripts/rst/pack_run.py --exp smoke
```

Send `runs/smoke.zip`.

## 2. Pilot (about an hour, roughly $10–20)

```bash
python scripts/rst/run_benchmark.py --exp pilot --datasets cadtestbench-detailed --limit 40 --methods zero_shot,react,tests_log,rst
python scripts/rst/pack_run.py --exp pilot
```

Send `runs/pilot.zip`. The pilot measures real token use per method, which replaces the estimates below, and
tells us whether the full runs are worth it (the go/no-go gates in the plan).

## 3. Main runs (only after the pilot is reviewed)

| Run | Command | Est. prompts × seeds × methods |
|---|---|---|
| CADTestBench, both splits | `--exp main_ctb --datasets cadtestbench-detailed,cadtestbench-abstract --seeds 0,1,2 --methods zero_shot,react,best_of_n,cadcodeverify,tests_log,rst` | 400 × 3 × 6 |
| Hard-Long | `--exp main_hl --datasets hardlong --seeds 0,1,2 --methods zero_shot,react,best_of_n,tests_log,rst` | 30 × 3 × 5 |
| Ablations | `--exp ablations --datasets cadtestbench-detailed --methods rst,rst_llm_localize,rst_random_localize,rst_whole_rewrite,rst_accept_always` | 200 × 1 × 5 |
| CADSmith (bug-fixed) | `--exp cadsmith --datasets cadsmith,cadtestbench-detailed --methods cadsmith_fixed,cadsmith_fixed_noleak,rst --workers 1` | 300 × 1 × 3 |
| Text2CAD subset | `--exp t2c --datasets text2cad --methods zero_shot,tests_log,rst` | 500 × 1 × 3 |
| E3: matrix vs LLM localisation | `python scripts/rst/e3_localization.py --dataset hardlong-all --spec entry --llm --n-mutants 8 --seed 1` (then `--seed 2`) | 70 × 8 mutants × 2 |
| E3 with self-written specs | `python scripts/rst/e3_localization.py --dataset hardlong-all --spec self --llm --n-mutants 8 --seed 1` | 70 × 8 mutants |
| E3 on CADTestBench | `python scripts/rst/e3_localization.py --dataset cadtestbench-detailed --spec oracle --threaded --llm --seed 1` (then `--seed 2`) | 200 × 5 mutants × 2 |

Prefix the benchmark rows (all but E3) with `python scripts/rst/run_benchmark.py`. Runs resume: if a run stops (for
example when session credentials expire), refresh `.env` and re-run the same command; finished results are skipped
and failed ones are retried. The E3 commands resume with `--resume` added (finished parts are kept, the part it
stopped in is redone). E3 uses mutant seeds 1 and 2 because seed 0 was used to develop the blame rules.
`cadsmith_fixed` must run with `--workers 1` (the CADSmith code keeps global counters).

## Open-weight generators (Colab GPU, no AWS)

To show that RST also improves trained text-to-CAD models, generate their programs on a GPU and let the
Bedrock runner start from them. On a Colab A100 (40 GB is enough for the 7B models):

```bash
git clone -b research_idea https://github.com/prashantkul366/CAD_AGENTIC.git
cd CAD_AGENTIC
pip install vllm pandas pyarrow cadquery==2.6.1
python scripts/data/fetch_cadtestbench.py
python scripts/data/fetch_text2cad.py --ids-file data/text2cad_subset_ids.txt
python scripts/rst/generate_open_model.py --model cad-coder --exp open_cadcoder --datasets cadtestbench-detailed,hardlong,text2cad
python scripts/rst/generate_open_model.py --model procad --exp open_procad --datasets cadtestbench-detailed,hardlong,text2cad
```

`fetch_text2cad.py` also downloads the file CAD-Coder's prompt template is read from. Zip `runs/open_cadcoder/p0`
(and `runs/open_procad/p0`) and unzip them into `runs/` on the Bedrock machine, then:

```bash
python scripts/rst/run_benchmark.py --exp open_cadcoder --datasets cadtestbench-detailed,hardlong,text2cad --methods zero_shot,tests_log,rst
```

`zero_shot` is then the trained model alone, and `rst` / `tests_log` wrap it (Claude writes the requirements and repairs).

## Cost and time estimates (replace with pilot measurements)

Per prompt and seed, with the default budget of 10 LLM calls per method (shared initial program and
requirement writer counted once):

| Method | Calls | Input tokens | Output tokens |
|---|---|---|---|
| zero_shot | 1 | ~0.3 K | ~0.6 K |
| react | ≤ 4 | ~6 K | ~2 K |
| tests_log | ≤ 10 | ~20 K | ~6 K |
| rst | ≤ 10 | ~20 K | ~3 K |

At Anthropic's first-party list price for Claude Sonnet 4.6 ($3 / $15 per million input / output tokens) that is
about $0.30 per prompt and seed for the four core methods. Amazon Bedrock prices differ; check
https://aws.amazon.com/bedrock/pricing/. Rough totals at list price: pilot ≈ $12; CADTestBench main ≈ $400;
Hard-Long ≈ $30; ablations ≈ $60; Text2CAD ≈ $120; CADSmith ≈ $30.

Speed: `--workers 4` runs four prompts in parallel; `RST_KERNEL_WORKERS` (default 4) sets how many CadQuery
processes run at once. Lower `--workers` if Bedrock throttles (HTTP 429). Expect roughly 1–2 minutes of wall
time per prompt per method with a 4-core machine.

## What to send back

For the E3 runs: `python scripts/rst/pack_run.py --exp e3` (zips `runs/e3/`).

`python scripts/rst/pack_run.py --exp <name>` writes `runs/<name>.zip` (configs, cached initial programs,
self-written requirements and every method's result; no bulky work files). Scoring happens on the receiving side:

```bash
python scripts/rst/evaluate_run.py --exp <name>
```
