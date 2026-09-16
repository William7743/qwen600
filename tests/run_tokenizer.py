"""Broader tokenizer differential regression, with reproducible Unicode/whitespace cases."""
import argparse
import json
import os
from pathlib import Path
import random
import struct
import subprocess

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = ["", "<", "<<", "<x", "a < b", "<not_special>", "x<|im_start|>user\nhello<|im_end|>",
             "<think>\n</think>", "\0", "a\0b", "e\u0301", "cafe\u0301", "\u0344", "\u0f73",
             "\u1100\u1161", "é", "你好，世界！🙂", "👩‍💻", "ＡＢＣ １２３", "İıſK",
             "I'm I'M we're WE'RE don't DON’T", "def add(a, b):\n    return a + b\n"]
    whitespace = [" ", "\t", "\n", "\r", "\r\n", "\v", "\f", "\u0085", "\u00a0", "\u1680",
                  "\u2003", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"]
    for ws in whitespace:
        for n in [1, 2, 3, 4, 8]:
            cases.extend([ws*n, "hello"+ws*n+"world", ws*n+"return x\n", "end"+ws*n])
    atoms = ["hello", "return", "foo", "123456", "3.14", "中国", "é", "e\u0301", "🙂", "_", "'s",
             "<", ">", "<|im_start|>", "</think>", "\u0f73", "\0"] + whitespace
    rng = random.Random(20260916)
    for _ in range(500):
        cases.append("".join(rng.choice(atoms) for _ in range(rng.randint(1, 24))))
    source = args.output / "tokenizer-corpus.bin"
    target = args.output / "tokenizer-corpus.ids.bin"
    with source.open("wb") as out:
        out.write(struct.pack("<I", len(cases)))
        for text in cases:
            raw = text.encode("utf-8")
            out.write(struct.pack("<I", len(raw))); out.write(raw)
    run = subprocess.run([str(args.probe.resolve()), str(args.model.resolve()), str(source), str(target)],
                         capture_output=True, text=True, timeout=120)
    (args.output / "tokenizer-corpus.log").write_text(run.stdout+run.stderr)
    if run.returncode:
        raise RuntimeError(run.stdout+run.stderr)
    failures = []
    with target.open("rb") as data:
        assert struct.unpack("<I", data.read(4))[0] == len(cases)
        for index, text in enumerate(cases):
            count = struct.unpack("<I", data.read(4))[0]
            actual = list(struct.unpack(f"<{count}I", data.read(count*4)))
            reference = tokenizer.encode(text, add_special_tokens=False)
            if actual != reference:
                failures.append({"index": index, "text": text, "native_ids": actual, "reference_ids": reference})
        assert data.read() == b""
    report = {"cases": len(cases), "passed": len(cases)-len(failures), "failures": failures}
    (args.output / "tokenizer-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"TOKENIZER CORPUS: {report['passed']}/{len(cases)} PASS")
    for failure in failures[:5]: print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
