"""Generate initial programs (P0) with open-weight text-to-CadQuery models — run on a GPU (e.g. Colab A100).

The output uses the same cache format as run_benchmark.py (runs/<exp>/p0/<dataset>/<id>_s<seed>.json), so the
Bedrock runner then starts every method from the trained model's program:
  zero_shot  = the trained model alone
  rst        = RST wrapped around the trained model (Claude writes requirements and repairs)

Models and their published prompt formats:
  cad-coder   gudo7208/CAD-Coder (NeurIPS 2025, Qwen2.5-7B): its CoT instruction + "description:", answer in
              \\boxed{```python ...```}, final solid in r  (instruction text read from the CAD-Coder dataset file)
  procad      BBexist/ProCAD-coder (ICML 2026, Qwen2.5-7B): system + user template from config/code_generation.py
  t2cq        ricemonster/qwen2.5-3B-SFT (Text-to-CadQuery): "### Instruction:\\n{text}\\n\\n### Response:\\n", greedy

Colab:
    !git clone -b research_idea https://github.com/prashantkul366/CAD_AGENTIC.git && cd CAD_AGENTIC
    !pip install vllm pandas pyarrow
    !python scripts/data/fetch_cadtestbench.py            # and/or fetch_text2cad.py --ids-file ...
    !python scripts/rst/generate_open_model.py --model cad-coder --exp open_cadcoder --datasets cadtestbench-detailed
    # then zip runs/open_cadcoder/p0 and unzip it into runs/<exp>/p0 on the Bedrock machine

Local check without a GPU:  python scripts/rst/generate_open_model.py --model cad-coder --exp dry_open --datasets hardlong --limit 2 --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rst.datasets import load

MODELS = {
    "cad-coder": "gudo7208/CAD-Coder",
    "procad": "BBexist/ProCAD-coder",
    "t2cq": "ricemonster/qwen2.5-3B-SFT",
}

PROCAD_SYSTEM = """You are an expert in CadQuery and 3D CAD modeling. You specialize in generating precise CadQuery Python code from natural language descriptions of 3D shapes.

Your task is to:
1. Analyze the provided text description of a 3D CAD model
2. Generate equivalent CadQuery Python code that creates the described shape
3. Ensure the code is correct, complete, and follows CadQuery best practices

Requirements:
- Start with: import cadquery as cq
- Store the final result in variable 'r'
- Use CadQuery operations only (no other libraries)
- Match the dimensions and features described in the text
- Output only the Python code, no explanations or markdown"""

PROCAD_USER = """Generate CadQuery Python code for the following 3D shape:

{description}

Output only the Python code, no explanations."""


def cad_coder_instruction(path: Path) -> str:
    """The instruction text CAD-Coder was trained with: everything before 'description:' in its test prompts."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/data/fetch_text2cad.py first (it downloads the file)")
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data:
        for m in e.get("messages") or []:
            if m.get("role") == "user" and "description:" in m.get("content", ""):
                return m["content"].split("description:")[0] + "description:\n"
    raise ValueError("no CAD-Coder instruction found")


def build_prompt(model_key: str, text: str, tokenizer, cad_coder_prefix: str = "") -> str:
    if model_key == "t2cq":
        return f"### Instruction:\n{text}\n\n### Response:\n"
    if model_key == "procad":
        messages = [{"role": "system", "content": PROCAD_SYSTEM},
                    {"role": "user", "content": PROCAD_USER.format(description=text)}]
    else:
        messages = [{"role": "user", "content": cad_coder_prefix + text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_code(raw: str) -> str:
    """Pull the program out of a model answer and make sure the final solid is named `result`."""
    text = raw
    m = re.search(r"\\boxed\{\s*```(?:python)?\s*\n(.*?)```\s*\}", text, re.S)
    if m:
        code = m.group(1)
    else:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        code = m.group(1) if m else re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    code = code.strip()
    if "### Response:" in code:
        code = code.split("### Response:", 1)[1].strip()
    if not re.search(r"^result\s*=", code, re.M) and re.search(r"^r\s*=", code, re.M):
        code += "\nresult = r\n"
    return code + ("\n" if not code.endswith("\n") else "")


class DryGenerator:
    def __init__(self):
        self.tokenizer = None

    def __call__(self, prompts, temperature, max_new_tokens, seed):
        box = '\\boxed{```python\nimport cadquery as cq\nr = cq.Workplane("XY").box(10, 10, 10)\n```}'
        return [("<think>plan</think>\n" + box, len(p) // 4, 40) for p in prompts]


class VLLMGenerator:
    def __init__(self, model_id: str, max_model_len: int):
        from vllm import LLM
        self.llm = LLM(model=model_id, dtype="bfloat16", max_model_len=max_model_len, trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()

    def __call__(self, prompts, temperature, max_new_tokens, seed):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=seed)
        outs = self.llm.generate(prompts, sp)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]


class HFGenerator:
    def __init__(self, model_id: str, max_model_len: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto",
                                                          trust_remote_code=True).eval()

    def __call__(self, prompts, temperature, max_new_tokens, seed):
        self.torch.manual_seed(seed)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=temperature if temperature > 0 else None,
                                      pad_token_id=self.tokenizer.pad_token_id)
        res = []
        for i in range(len(prompts)):
            gen = out[i][enc["input_ids"].shape[1]:]
            res.append((self.tokenizer.decode(gen, skip_special_tokens=True), int(enc["attention_mask"][i].sum()), len(gen)))
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="cad-coder | procad | t2cq | a Hugging Face model id")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets", default="cadtestbench-detailed")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: greedy for seed 0 (as published), 0.7 for other seeds")
    ap.add_argument("--cad-coder-template", default=str(ROOT / "data/external/text2cad/cad_data_test_cot.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = args.model if args.model in MODELS else ("t2cq" if "3B-SFT" in args.model else "procad")
    model_id = MODELS.get(args.model, args.model)
    if args.dry_run:
        gen = DryGenerator()
    elif args.backend == "vllm":
        gen = VLLMGenerator(model_id, args.max_model_len)
    else:
        gen = HFGenerator(model_id, args.max_model_len)
    prefix = cad_coder_instruction(Path(args.cad_coder_template)) if key == "cad-coder" and not args.dry_run else ""
    exp = ROOT / "runs" / args.exp
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        entries = load(ds, args.limit)
        for seed in [int(s) for s in args.seeds.split(",")]:
            temp = args.temperature if args.temperature is not None else (0.0 if seed == 0 else 0.7)
            todo = [e for e in entries if not (exp / "p0" / ds / f"{e['id']}_s{seed}.json").exists()]
            print(f"{ds} seed {seed}: {len(todo)} prompts to generate with {model_id}", flush=True)
            for i in range(0, len(todo), args.batch):
                batch = todo[i:i + args.batch]
                if args.dry_run:
                    prompts = [e["prompt"] for e in batch]
                else:
                    prompts = [build_prompt(key, e["prompt"], gen.tokenizer, prefix) for e in batch]
                outs = gen(prompts, temp, args.max_new_tokens, seed)
                for e, (raw, n_in, n_out) in zip(batch, outs):
                    path = exp / "p0" / ds / f"{e['id']}_s{seed}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({
                        "code": extract_code(raw), "raw": raw, "generator": model_id, "temperature": temp,
                        "usage": {"calls": 1, "input_tokens": n_in, "output_tokens": n_out, "by_role": {}},
                    }), encoding="utf-8")
                print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
    (exp / "p0_generator.json").write_text(json.dumps({"model": model_id, "key": key, "args": vars(args)}, indent=2),
                                           encoding="utf-8")
    print("done ->", exp / "p0")


if __name__ == "__main__":
    main()
