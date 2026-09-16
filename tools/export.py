# export.py

import argparse
import json
import os
import struct
from pathlib import Path

from jinja2 import Template

def bytes_to_unicode():
    """Reference GPT-2 byte→Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )

def build_tokenizer(model, output_dir):
    """Export exact byte IDs, added tokens, regex and ranked BPE pairs (QTK2)."""
    tokenizer = model.tokenizer
    data = json.loads(tokenizer.backend_tokenizer.to_str())
    bpe = data["model"]
    if bpe["type"] != "BPE" or bpe.get("dropout") or bpe.get("ignore_merges"):
        raise ValueError("Expected deterministic Qwen byte-level BPE")
    normalizer = data.get("normalizer")
    if normalizer not in (None, {"type": "NFC"}):
        raise ValueError("Only NFC or no normalization is supported")
    pre = data["pre_tokenizer"]
    if pre.get("type") != "Sequence" or len(pre["pretokenizers"]) != 2:
        raise ValueError("Expected Qwen Split + ByteLevel pre-tokenizer")
    split, bytelevel = pre["pretokenizers"]
    if (split.get("type") != "Split" or split.get("behavior") != "Isolated"
            or split.get("invert") or "Regex" not in split["pattern"]
            or bytelevel.get("type") != "ByteLevel"
            or bytelevel.get("add_prefix_space") or bytelevel.get("use_regex")):
        raise ValueError("Unsupported pre-tokenization configuration")
    pattern = split["pattern"]["Regex"].encode("utf-8")
    added = data["added_tokens"]
    for token in added:
        if any(token.get(k) for k in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("Unsupported added-token matching flags")
        if not token["content"]:
            raise ValueError("Empty added token")
    added_by_id = {t["id"]: t["content"] for t in added}
    vocab = tokenizer.get_vocab()
    id_to_token = {index: token for token, index in vocab.items()}
    count = max(id_to_token) + 1
    if set(id_to_token) != set(range(count)):
        raise ValueError("Vocabulary IDs must be contiguous")
    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    base_vocab = bpe["vocab"]
    merges = []
    for merge in bpe["merges"]:
        left, right = merge if isinstance(merge, list) else merge.split()
        merges.append((base_vocab[left], base_vocab[right], base_vocab[left + right]))
    path = Path(output_dir) / "tokenizer.bin"
    temporary = path.with_suffix(".bin.tmp")
    try:
        with temporary.open("wb") as out:
            out.write(b"QTK2")
            out.write(struct.pack("<7I", count, model.bos_token_id, model.eos_token_id,
                                  len(merges), len(added), len(pattern), int(normalizer is not None)))
            out.write(pattern)
            for index in range(count):
                # Added tokens are literal text, not GPT-2 byte-encoded vocabulary strings.
                value = (added_by_id[index].encode("utf-8") if index in added_by_id
                         else internal_to_bytes(u2b, id_to_token[index]))
                out.write(struct.pack("<I", len(value)))
                out.write(value)
            out.write(struct.pack("<256I", *(base_vocab[b2u[b]] for b in range(256))))
            for token in added:
                out.write(struct.pack("<I", token["id"]))
            for merge in merges:
                out.write(struct.pack("<3I", *merge))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Written QTK2 tokenizer to {path} ({count} tokens, {len(merges)} merges)")

def build_prompts(model, output_dir):
    template = Template(model.tokenizer.chat_template)

    # Template 1: User
    messages = [{"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_user.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 2: User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_user_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 3: System + User
    messages = [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_system.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 4: System + User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_system_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    print(f"Written prompt templates to '{output_dir}'")

# -----------------------------------------------------------------------------
# Load / import functions

def load_tokenizer_and_config(model_path):
    """Loads only the tokenizer and config, not the full model weights."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        from types import SimpleNamespace
    except ImportError:
        print("Error: transformers package is required.")
        print("Please run `pip install transformers` to install it.")
        return None

    print(f"Loading tokenizer and config from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)

    model_mock = SimpleNamespace()
    model_mock.tokenizer = tokenizer
    model_mock.bos_token_id = hf_config.bos_token_id if hasattr(hf_config, "bos_token_id") else 0
    model_mock.eos_token_id = hf_config.eos_token_id if hasattr(hf_config, "eos_token_id") else 0
    
    print("Successfully loaded tokenizer and config.")
    return model_mock

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tokenizer.bin and template files")
    parser.add_argument("model_path", type=str, help="Path to the local Hugging Face model directory (used for both input and output).")
    args = parser.parse_args()

    model_info = load_tokenizer_and_config(args.model_path)

    if model_info:
        build_tokenizer(model_info, args.model_path)
        build_prompts(model_info, args.model_path)
