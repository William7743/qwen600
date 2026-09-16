"""Compare production CUDA/tokenizer code with a local Transformers reference.

No weights or inference code are modified. A failed structural check exits 1.
Logit differences are diagnostics, not an asserted universal BF16 tolerance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_CONTEXT_TOKENS = 1024
MAX_NEW_TOKENS = 24


def command(args):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def read_ids(path):
    return [int(v) for v in path.read_text().split()]


def metrics(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"finite": False}
    pa = np.exp(a-a.max()); pa /= pa.sum()
    pb = np.exp(b-b.max()); pb /= pb.sum()
    order = np.argsort(b)
    return {
        "finite": True,
        "mean_absolute_error": float(np.abs(a-b).mean()),
        "max_absolute_error": float(np.abs(a-b).max()),
        "cosine_similarity": float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))),
        "probability_total_variation": float(np.abs(pa-pb).sum()/2),
        "native_top1": int(a.argmax()), "reference_top1": int(b.argmax()),
        "top1_equal": bool(a.argmax() == b.argmax()),
        "reference_top1_margin": float(b[order[-1]]-b[order[-2]]),
        "top10_overlap": len(set(np.argsort(a)[-10:]) & set(order[-10:])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.model = args.model.resolve(); args.probe = args.probe.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = {"environment": {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "git_commit": command(["git", "rev-parse", "HEAD"]).strip(),
        "git_status": command(["git", "status", "--short"]),
        "model_path": str(args.model), "dtype": "bfloat16", "attention": "eager",
        "reference_forward": "one token per step, KV cache enabled",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "scope": "prompt plus generated tokens <= 1024; no out-of-range attention tests",
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((args.model / "tokenizer.bin").read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path("engine/main.cu"), Path("models/qwen_model.cuh"), Path("utils/tokenizer.h"),
             Path("layers/sampler.h"), Path("tools/export.py"), Path("tests/correctness_probe.cu")]},
    }, "tokenization": [], "templates": [], "forward": [], "generation": []}
    with (args.model / "model.safetensors").open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(8*1024*1024), b""): digest.update(chunk)
    report["environment"]["model_sha256"] = digest.hexdigest()

    prompts = [
        ("english", "What is the capital of France? Answer in one short sentence."),
        ("chinese", "中国的首都是哪里？请用一句话回答。"),
        ("arithmetic", "What is 2 + 3? Answer with only the number."),
        ("code", "def add(a, b):\n    return a + b\n"),
        ("unicode", "你好，世界！🙂 café naïve"),
        ("digits", "1234567890 2026-09-16 3.1415926535"),
        ("whitespace", "hello   world\n\n\tend  "),
        ("contractions", "I'm testing: don't, we're, IT'S."),
    ]
    cases = []
    for thinking in [False, True]:
        for system in [False, True]:
            messages = ([{"role": "system", "content": "You are a helpful assistant."}] if system else [])
            messages += [{"role": "user", "content": prompts[0][1]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            name = "template_" + ("system" if system else "user") + ("_thinking" if thinking else "")
            template = (args.model / (name+".txt")).read_text()
            content = ("You are a helpful assistant.", prompts[0][1]) if system else prompts[0][1]
            report["templates"].append({"name": name, "equal": template % content == text})
            cases.append((name, text))
    cases += prompts
    for name, text in cases:
        src = args.output / (name+".txt"); src.write_text(text)
        dest = args.output / (name+".native.ids")
        command([args.probe, "tokenize", args.model, src, dest])
        actual = read_ids(dest); reference = tokenizer.encode(text, add_special_tokens=False)
        report["tokenization"].append({"name": name, "equal": actual == reference,
            "native_ids": actual, "reference_ids": reference,
            "decode_roundtrip": tokenizer.decode(actual) == text})
        print("TOKENIZER", name, "PASS" if actual == reference else "FAIL", flush=True)

    boundary = subprocess.run([str(args.probe), "attention"], capture_output=True, text=True, timeout=60)
    if boundary.returncode not in (0, 1): raise RuntimeError(boundary.stderr)
    report["attention_boundary"] = [json.loads(line) for line in boundary.stdout.splitlines()]
    print("ATTENTION", report["attention_boundary"], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager").eval().to("cuda")

    @torch.inference_mode()
    def step(token, cache):
        result = model(input_ids=torch.tensor([[token]], device="cuda"),
                       past_key_values=cache, use_cache=True)
        return result.logits[0, -1].float().cpu().numpy(), result.past_key_values

    for name, prompt in prompts[:3]:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + MAX_NEW_TOKENS > MAX_CONTEXT_TOKENS:
            raise ValueError(f"{name}: prompt plus generation exceeds 1024 tokens")
        src = args.output / (name+".reference.ids"); src.write_text("\n".join(map(str, ids)))
        positions = list(range(len(ids)))
        dest = args.output / (name+".logits.f32")
        command([args.probe, "forward", args.model, src, dest, " ".join(map(str, positions))])
        actual = np.fromfile(dest, dtype=np.float32).reshape(len(ids), model.config.vocab_size)
        cache = None; rows = []
        for pos, token in enumerate(ids):
            logits, cache = step(token, cache)
            rows.append({"position": pos, **metrics(actual[pos], logits)})
        report["forward"].append({"name": name, "input_tokens": len(ids), "positions": rows})
        print("FORWARD", name, "top1", sum(r.get("top1_equal", False) for r in rows), "/", len(rows), flush=True)
        generation = []; teacher_ids = ids.copy()
        for i in range(MAX_NEW_TOKENS):
            token = int(logits.argmax()); generation.append(token)
            if token in (151645, 151643): break
            if i+1 < MAX_NEW_TOKENS: logits, cache = step(token, cache)
        dest = args.output / (name+".greedy.ids")
        command([args.probe, "greedy", args.model, src, dest, MAX_NEW_TOKENS])
        native = read_ids(dest)
        # Teacher forcing isolates per-step numeric differences from divergent histories.
        teacher_ids += generation[:-1]
        src = args.output / (name+".teacher.ids"); src.write_text("\n".join(map(str, teacher_ids)))
        start = len(ids)-1; positions = list(range(start, len(teacher_ids)))
        dest = args.output / (name+".teacher.f32")
        command([args.probe, "forward", args.model, src, dest, " ".join(map(str, positions))])
        teacher_native = np.fromfile(dest, dtype=np.float32).reshape(len(positions), model.config.vocab_size).argmax(axis=1).tolist()
        report["generation"].append({"name": name, "equal": native == generation,
            "native_ids": native, "reference_ids": generation,
            "native_text": tokenizer.decode(native), "reference_text": tokenizer.decode(generation),
            "teacher_forced_native_top1": teacher_native,
            "teacher_forced_matches": sum(a == b for a, b in zip(teacher_native, generation))})
        print("GENERATION", name, "PASS" if native == generation else "DIFF", flush=True)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Use identical fixed IDs, bypassing tokenization, up to the supported boundary.
    seed = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)
    ids = (seed * (MAX_CONTEXT_TOKENS//len(seed)+1))[:MAX_CONTEXT_TOKENS]
    positions = [127, 511, 1022, 1023]
    src = args.output / "boundary.ids"; src.write_text("\n".join(map(str, ids)))
    dest = args.output / "boundary.logits.f32"
    command([args.probe, "forward", args.model, src, dest, " ".join(map(str, positions))])
    actual = np.fromfile(dest, dtype=np.float32).reshape(len(positions), model.config.vocab_size)
    rows = []; cache = None
    for pos, token in enumerate(ids):
        logits, cache = step(token, cache)
        if pos in positions:
            row = {"position": pos, **metrics(actual[positions.index(pos)], logits)}
            rows.append(row)
            print("BOUNDARY", pos, row, flush=True)
    report["forward"].append({"name": "boundary", "input_tokens": len(ids), "positions": rows})
    failures = sum(not r["equal"] for r in report["tokenization"]+report["templates"])
    failures += sum(r["incorrect_scores"] > 0 for r in report["attention_boundary"])
    failures += sum(not row["finite"] for case in report["forward"] for row in case["positions"])
    report["structural_failures"] = failures
    generation_failures = sum(not case["equal"] or
        case["teacher_forced_matches"] != len(case["reference_ids"]) for case in report["generation"])
    report["generation_failures"] = generation_failures
    report["verdict"] = "FAIL" if failures or generation_failures else "STRUCTURAL_CHECKS_PASS_NUMERICAL_REVIEW_REQUIRED"
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("Report:", args.output / "report.json", "verdict:", report["verdict"], flush=True)
    return 1 if failures or generation_failures else 0


if __name__ == "__main__":
    sys.exit(main())
