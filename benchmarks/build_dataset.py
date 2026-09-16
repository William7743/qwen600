"""Build or verify the offline, fixed-workload batch=1 benchmark dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from importlib.metadata import version

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
LENGTHS = (16, 256, 1024)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Verify without writing files')
    args = parser.parse_args()
    lock = json.loads((ROOT.parent / 'tests/reference.json').read_text())
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if digest((args.model / name).read_bytes()) != lock['model']['sha256'][name]:
            raise RuntimeError(f'Model tokenizer hash mismatch: {name}')
    versions = {name: version(name) for name in ('transformers', 'tokenizers', 'jinja2')}
    for name, actual in versions.items():
        if actual != lock['python_reference'][name]:
            raise RuntimeError(f'{name}: expected {lock["python_reference"][name]}, got {actual}')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = (ROOT / 'source.txt').read_text()
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    files = {}
    inputs = []
    for length in LENGTHS:
        # Trim only user content; always preserve the complete chat template.
        for count in range(min(len(source_ids), length), -1, -1):
            user = tokenizer.decode(source_ids[:count], clean_up_tokenization_spaces=False)
            rendered = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) == length:
                break
        else:
            raise RuntimeError(f'Cannot produce exact length {length} without breaking template')
        assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == rendered
        paths = {'user': f'inputs/p{length}.user.txt',
                 'prompt': f'inputs/p{length}.prompt.txt', 'ids': f'inputs/p{length}.ids'}
        files[paths['user']] = user.encode()
        files[paths['prompt']] = rendered.encode()
        files[paths['ids']] = ''.join(f'{token}\n' for token in ids).encode()
        inputs.append({'name': f'p{length}', 'prompt_tokens': length, 'files': paths,
                       'sha256': {key: digest(files[path]) for key, path in paths.items()}})
    manifest = {
        'schema_version': 1, 'dataset': 'qwen600-static-batch1-v1',
        'purpose': 'Performance workload, not a correctness or quality dataset',
        'source_sha256': digest(source.encode()), 'model': lock['model'],
        'tokenizer_versions': versions, 'batch_size': 1, 'concurrent_requests': 1,
        'chat_template': {'enable_thinking': False, 'add_generation_prompt': True},
        'sampling': {'mode': 'greedy', 'ignore_eos': True, 'fixed_output_tokens': True},
        'execution': {'load_model_once': True, 'reset_sequence_each_request': True,
                      'warmup_requests_per_case': 2, 'measured_requests_per_case': 5,
                      'print_tokens_during_timing': False, 'case_order': 'prompt_then_output'},
        'inputs': inputs,
        'cases': [{'id': f'p{p}_g{g}', 'input': f'p{p}', 'prompt_tokens': p,
                   'output_tokens': g, 'decode_forward_steps': g-1,
                   'total_tokens': p+g} for p in LENGTHS for g in LENGTHS],
    }
    files['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
    for name, data in files.items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise RuntimeError(f'Dataset mismatch: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(('VERIFIED' if args.check else 'BUILT') + ': 3 inputs (16/256/1024), 9 cases; full templates, exact token counts')


if __name__ == '__main__':
    main()
