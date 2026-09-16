"""Freeze 100 readable synthetic requests with seeded random lengths, all <1024."""
import argparse
import json
import random
from pathlib import Path
from importlib.metadata import version
from build_dataset import ROOT, digest, AutoTokenizer

SEED = 20260916
OUT = ROOT / 'diverse100'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT.parent/'tests/reference.json').read_text())
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if digest((args.model/name).read_bytes()) != lock['model']['sha256'][name]:
            raise RuntimeError(f'Tokenizer hash mismatch: {name}')
    versions = {n: version(n) for n in ('transformers', 'tokenizers', 'jinja2')}
    if any(v != lock['python_reference'][n] for n, v in versions.items()):
        raise RuntimeError('Tokenizer dependency version mismatch')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    topics = json.loads((ROOT/'diverse_topics.json').read_text())
    rng = random.Random(SEED)
    files, inputs, cases = {}, [], []
    def render(user):
        text = tok.apply_chat_template([{'role':'user','content':user}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        return text, tok.encode(text, add_special_tokens=False)
    for index in range(100):
        topic = topics[index % len(topics)]
        # Keep the complete task instruction; only the supporting material is trimmed.
        prefix = topic['instruction']+'\n'
        minimum = max(32, len(render(prefix)[1])+8)
        target, output = rng.randint(minimum, 1023), rng.randint(2, 1023)
        name = f'd{index+1:03d}'
        source = prefix + '\n'.join(
            (f'Record {index+1}.{j}: ' if topic['language']=='en' else f'记录 {index+1}.{j}：')
            + topic['body'] for j in range(1, 33))
        source_ids = tok.encode(source, add_special_tokens=False)
        found = None
        for count in range(target, 0, -1):
            user = tok.decode(source_ids[:count], clean_up_tokenization_spaces=False)
            if '\ufffd' in user or not user.startswith(prefix):
                continue
            text, ids = render(user)
            if len(ids) == target:
                found = user, text, ids; break
            if len(ids) < target:
                # A UTF-8 character can span several tokens. Add readable punctuation
                # only if necessary to reach the exact seeded target without replacement chars.
                for suffix in ('.', '。', '\n', ' ...', '。\n'):
                    text, ids = render(user+suffix)
                    if len(ids) == target:
                        found = user+suffix, text, ids; break
                if found:
                    break
        if found is None:
            raise RuntimeError(f'Cannot fit exact length: {name} {target}')
        user, text, ids = found
        assert tok.decode(ids, clean_up_tokenization_spaces=False) == text
        paths = {'user':f'inputs/{name}.user.txt', 'prompt':f'inputs/{name}.prompt.txt',
                 'ids':f'inputs/{name}.ids'}
        for key, data in [('user',user), ('prompt',text), ('ids',''.join(f'{i}\n' for i in ids))]:
            files[paths[key]] = data.encode()
        inputs.append({'name':name, 'topic':topic['topic'], 'language':topic['language'],
                       'prompt_tokens':target, 'files':paths,
                       'sha256':{k:digest(files[v]) for k,v in paths.items()}})
        cases.append({'id':name, 'input':name, 'prompt_tokens':target, 'output_tokens':output,
                      'decode_forward_steps':output-1, 'total_tokens':target+output})
    manifest = {'schema_version':1, 'dataset':'qwen600-diverse100-v1',
                'purpose':'Fixed synthetic performance workload, not a quality benchmark',
                'seed':SEED, 'length_sampling':{'prompt':'uniform integer [max(32, instruction+template tokens+8),1023]',
                    'output':'uniform integer [2,1023]', 'resampled_at_runtime':False},
                'topics_sha256':digest((ROOT/'diverse_topics.json').read_bytes()),
                'model':lock['model'], 'tokenizer_versions':versions, 'batch_size':1,
                'concurrent_requests':1, 'chat_template':{'enable_thinking':False,'add_generation_prompt':True},
                'sampling':{'mode':'greedy','ignore_eos':True,'fixed_output_tokens':True},
                'execution':{'warmup_requests_per_case':2,'measured_requests_per_case':5,
                    'load_model_once':True,'reset_sequence_each_request':True,'case_order':'d001..d100'},
                'inputs':inputs,'cases':cases}
    files['manifest.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    for name, data in files.items():
        path = OUT/name
        if args.check:
            if not path.exists() or path.read_bytes()!=data:
                raise RuntimeError(f'Dataset mismatch: {path}')
        else:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    print(('VERIFIED' if args.check else 'BUILT')+f': 100 cases, seed={SEED}; prompt range '
          f'{min(c["prompt_tokens"] for c in cases)}..{max(c["prompt_tokens"] for c in cases)}, output range '
          f'{min(c["output_tokens"] for c in cases)}..{max(c["output_tokens"] for c in cases)}')


if __name__=='__main__':
    main()
