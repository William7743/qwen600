"""Build the frozen ShareGPT subset offline from the checked-in first-pair snapshot."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from build_dataset import ROOT, digest, AutoTokenizer

OUT = ROOT/'sharegpt100'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    lock = json.loads((ROOT.parent/'tests/reference.json').read_text())
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if digest((args.model/name).read_bytes()) != lock['model']['sha256'][name]:
            raise RuntimeError(f'Tokenizer hash mismatch: {name}')
    versions = {n:version(n) for n in ('transformers','tokenizers','jinja2')}
    if any(v != lock['python_reference'][n] for n,v in versions.items()):
        raise RuntimeError('Tokenizer dependency version mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source_path = OUT/'source_pairs.json'
    source = json.loads(source_path.read_text())
    files, inputs, cases, seen = {}, [], [], set()
    skipped = Counter()
    for record in source['records']:
        turns = record['first_two_turns']
        if len(turns) != 2 or turns[0].get('from') != 'human' or turns[1].get('from') != 'gpt':
            skipped['roles'] += 1; continue
        prompt, reply = turns[0].get('value'), turns[1].get('value')
        if not isinstance(prompt,str) or not isinstance(reply,str) or not prompt.strip() or not reply.strip():
            skipped['empty_or_nontext'] += 1; continue
        rendered = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        output = len(tokenizer.encode(reply, add_special_tokens=False))
        if not (0 < len(ids) < 1024 and 2 <= output < 1024):
            skipped['length'] += 1; continue
        ids_bytes = ''.join(f'{t}\n' for t in ids).encode()
        if digest(ids_bytes) in seen:
            skipped['duplicate_input'] += 1; continue
        seen.add(digest(ids_bytes))
        name = f's{len(cases)+1:03d}'
        paths = {'user':f'inputs/{name}.user.txt', 'prompt':f'inputs/{name}.prompt.txt',
                 'ids':f'inputs/{name}.ids', 'reference_reply':f'inputs/{name}.reply.txt'}
        for key,data in [('user',prompt.encode()),('prompt',rendered.encode()),
                         ('ids',ids_bytes),('reference_reply',reply.encode())]:
            files[paths[key]] = data
        inputs.append({'name':name, 'source_id':record['id'], 'source_index':record['index'],
                       'prompt_tokens':len(ids), 'files':paths,
                       'sha256':{k:digest(files[v]) for k,v in paths.items()}})
        cases.append({'id':name,'input':name,'prompt_tokens':len(ids),'output_tokens':output,
                      'decode_forward_steps':output-1,'total_tokens':len(ids)+output})
        if len(cases)==100:
            break
    if len(cases)!=100:
        raise RuntimeError(f'Insufficient qualifying pairs: {len(cases)}')
    manifest = {'schema_version':1,'dataset':'qwen600-sharegpt100-v1',
        'purpose':'Filtered single-turn performance workload; original replies specify length only',
        'source':source['provenance'],'source_pairs_sha256':digest(source_path.read_bytes()),
        'selection':'First 100 unique qualifying first human/gpt pairs in pinned source order; no truncation',
        'skipped_before_selection_complete':dict(skipped),
        'model':lock['model'],'tokenizer_versions':versions,'batch_size':1,'concurrent_requests':1,
        'chat_template':{'enable_thinking':False,'add_generation_prompt':True},
        'sampling':{'mode':'greedy','ignore_eos':True,'fixed_output_tokens':True},
        'execution':{'global_warmup_requests':1,'measured_requests_per_case':1,
            'warmup_selection':'closest to P=512,G=512; tie by case id',
            'load_model_once':True,'reset_sequence_each_request':True,'case_order':'s001..s100'},
        'inputs':inputs,'cases':cases}
    files['manifest.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    for name,data in files.items():
        path = OUT/name
        if args.check:
            if not path.exists() or path.read_bytes()!=data:
                raise RuntimeError(f'Dataset mismatch: {path}')
        else:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    print(('VERIFIED' if args.check else 'BUILT')+': 100 ShareGPT pairs; prompt '
          f'{min(c["prompt_tokens"] for c in cases)}..{max(c["prompt_tokens"] for c in cases)}, output '
          f'{min(c["output_tokens"] for c in cases)}..{max(c["output_tokens"] for c in cases)}')


if __name__=='__main__':
    main()
