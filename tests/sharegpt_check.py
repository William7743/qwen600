"""ShareGPT-only logits acceptance. Freeze once on trusted V0; check any candidate."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from check_optimization import logit_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'benchmarks/sharegpt100'
VOCAB = 151936
FROZEN_V0 = 'b5c18694aa7ab2524974d32b142fa500149f41f2'
PIN = ROOT/'tests/sharegpt_reference.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fresh(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError('Directory must be empty; refusing to overwrite: '+str(path))
    path.mkdir(parents=True, exist_ok=True)


def run(command, log):
    with log.open('w') as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=3600)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def validate_model(model, expected):
    for name, digest in expected.items():
        if sha(model/name) != digest:
            raise ValueError('Model hash mismatch: '+name)


def positions(prompt, output):
    # Prompt start/middle/end, then first/middle/last decode prediction.
    return sorted({0, (prompt-1)//2, prompt-1, prompt,
                   prompt+(output-2)//2, prompt+output-2})


def build_plan(cases, base, path):
    path.write_text(''.join(
        f'{json.dumps(str((base/c["history_file"]).resolve()))} {c["prompt_tokens"]} '
        f'{len(c["positions"])} '+ ' '.join(map(str, c['positions']))+'\n'
        for c in cases))


def freeze(args):
    # Only the historical V0 production sources may provide the reference.
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', FROZEN_V0,
                                     'models', 'layers', 'utils', 'config.h'], cwd=ROOT, text=True).splitlines()
    if not files:
        raise ValueError('Frozen V0 sources missing')
    source_hashes = {}
    for name in files:
        original = subprocess.check_output(['git', 'show', FROZEN_V0+':'+name], cwd=ROOT)
        if (ROOT/name).read_bytes() != original:
            raise ValueError('Freeze requires unchanged V0 production sources: '+name)
        source_hashes[name] = sha(ROOT/name)
    cache = args.build_dir/'CMakeCache.txt'
    if f'CMAKE_HOME_DIRECTORY:INTERNAL={ROOT}' not in cache.read_text().splitlines():
        raise ValueError('Build directory does not belong to this V0 checkout')
    fresh(args.baseline)
    print('Building trusted V0 probes...', flush=True)
    run(['cmake', '--build', str(args.build_dir), '--target', 'benchmark_probe', 'iteration_probe', '-j', '4'],
        args.baseline/'build.log')
    # Generate real V0 continuations. The original ShareGPT reply specifies G only.
    print('Generating 100 fixed V0 continuations...', flush=True)
    run([sys.executable, str(ROOT/'benchmarks/run_benchmark.py'), '--model', str(args.model),
         '--build-dir', str(args.build_dir), '--dataset', 'sharegpt100', '--warmup', '0',
         '--output', str(args.baseline/'generation')], args.baseline/'generation.log')
    dataset = read(DATA/'manifest.json')
    raw = [json.loads(line) for line in (args.baseline/'generation/raw.jsonl').read_text().splitlines()]
    generated = {r['case']: r for r in raw if r['event']=='request' and not r['warmup']}
    inputs = {x['name']: x for x in dataset['inputs']}
    histories = args.baseline/'histories'
    histories.mkdir()
    cases = []
    offset = 0
    for c in dataset['cases']:
        row = generated[c['id']]
        prompt = list(map(int, (DATA/inputs[c['input']]['files']['ids']).read_text().split()))
        ids = prompt+row['generated_ids'][:-1]
        file = 'histories/'+c['id']+'.ids'
        (args.baseline/file).write_text(''.join(str(x)+'\n' for x in ids))
        selected = positions(len(prompt), c['output_tokens'])
        cases.append({'id': c['id'], 'prompt_tokens': len(prompt), 'output_tokens': c['output_tokens'],
                      'history_file': file, 'history_sha256': sha(args.baseline/file),
                      'positions': selected, 'offset': offset})
        offset += len(selected)
    ranked = sorted(cases, key=lambda c: (c['prompt_tokens']+c['output_tokens'], c['id']))
    quick = [ranked[i]['id'] for i in (0, 19, 39, 59, 79, 99)]
    plan = args.baseline/'plan.txt'
    build_plan(cases, args.baseline, plan)
    print(f'Capturing {offset} full-vocabulary logits vectors...', flush=True)
    run([str(args.build_dir/'bin/iteration_probe'), str(args.model/'model.safetensors'),
         str(plan), str(args.baseline/'logits.f32'), 'logits'], args.baseline/'native.log')
    if (args.baseline/'logits.f32').stat().st_size != offset*VOCAB*4:
        raise ValueError('Incomplete reference logits')
    ref = np.memmap(args.baseline/'logits.f32', mode='r', dtype='<f4', shape=(offset,VOCAB))
    if any(not np.isfinite(x).all() for x in ref):
        raise ValueError('Nonfinite reference')
    manifest = {'version': 1, 'scope': 'ShareGPT V0-relative sampled logits; not HF certification',
                'v0_source_commit': FROZEN_V0, 'source_hashes': source_hashes,
                'iteration_probe_sha256': sha(args.build_dir/'bin/iteration_probe'),
                'benchmark_probe_sha256': sha(args.build_dir/'bin/benchmark_probe'),
                'dataset_manifest_sha256': sha(DATA/'manifest.json'),
                'model_sha256': dataset['model']['sha256'],
                'logits_limits': read(ROOT/'tests/optimization_policy.json')['logits'],
                'vocab_size': VOCAB, 'snapshots': offset, 'cases': cases, 'quick_cases': quick,
                'logits_sha256': sha(args.baseline/'logits.f32')}
    write(args.baseline/'manifest.json', manifest)
    write(PIN, {'version': 1, 'manifest_sha256': sha(args.baseline/'manifest.json'),
                'cases': len(cases), 'snapshots': offset, 'quick_cases': quick,
                'v0_source_commit': FROZEN_V0})
    print('FROZEN', args.baseline, 'manifest SHA256', sha(args.baseline/'manifest.json'))
    print('Keep this reference package and its pin fixed throughout optimization:', PIN)


def check(args):
    started = time.perf_counter()
    pin = read(PIN)
    if not pin.get('manifest_sha256'):
        raise ValueError('Reference not initialized: run freeze on unchanged V0 before optimizing')
    if sha(args.baseline/'manifest.json') != pin['manifest_sha256']:
        raise ValueError('Reference manifest differs from the locally frozen V0 hash')
    manifest = read(args.baseline/'manifest.json')
    policy = read(ROOT/'tests/optimization_policy.json')['logits']
    if manifest['logits_limits'] != policy or manifest['dataset_manifest_sha256'] != sha(DATA/'manifest.json'):
        raise ValueError('Policy or ShareGPT dataset differs from frozen reference')
    validate_model(args.model, manifest['model_sha256'])
    if sha(args.baseline/'logits.f32') != manifest['logits_sha256']:
        raise ValueError('Reference logits hash mismatch')
    if (args.baseline/'logits.f32').stat().st_size != manifest['snapshots']*VOCAB*4:
        raise ValueError('Reference logits size mismatch')
    cases = [c for c in manifest['cases'] if not args.quick or c['id'] in manifest['quick_cases']]
    for c in cases:
        if sha(args.baseline/c['history_file']) != c['history_sha256']:
            raise ValueError('Reference history modified: '+c['id'])
    fresh(args.output)
    plan = args.output/'plan.txt'
    build_plan(cases, args.baseline, plan)
    probe = args.build_dir/'bin/iteration_probe'
    print(f'Checking {len(cases)} ShareGPT histories; full vocabulary at selected positions...', flush=True)
    run([str(probe), str(args.model/'model.safetensors'), str(plan), str(args.output/'actual.f32'), 'logits'],
        args.output/'native.log')
    count = sum(len(c['positions']) for c in cases)
    if (args.output/'actual.f32').stat().st_size != count*VOCAB*4:
        raise ValueError('Missing or extra candidate logits')
    actual = np.memmap(args.output/'actual.f32', mode='r', dtype='<f4', shape=(count,VOCAB))
    ref = np.memmap(args.baseline/'logits.f32', mode='r', dtype='<f4', shape=(manifest['snapshots'],VOCAB))
    rows = []
    index = 0
    for c in cases:
        failures = 0
        for j, pos in enumerate(c['positions']):
            a, b = ref[c['offset']+j], actual[index]
            index += 1
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            metrics = logit_metrics(a,b) if finite else {}
            passed = finite and all(metrics[k.removesuffix('_max')] <= value for k,value in policy.items())
            failures += not passed
            rows.append({'case': c['id'], 'position': pos, 'pass': passed, 'finite': finite, **metrics})
            if not passed:
                print('FAIL', c['id'], 'position', pos, metrics, flush=True)
        print(c['id'], 'PASS' if not failures else 'FAIL', flush=True)
    failed = [r for r in rows if not r['pass']]
    report = {'status': 'FAIL' if failed else 'PASS', 'quick': args.quick,
              'scope': manifest['scope'], 'cases': len(cases), 'checks': len(rows),
              'baseline_manifest_sha256': pin['manifest_sha256'], 'probe_sha256': sha(probe),
              'logits_limits': policy, 'failures': failed, 'results': rows,
              'max_errors': {k: max((r.get(k.removesuffix('_max'), 0) for r in rows), default=0)
                             for k in policy} if all(r['finite'] for r in rows) else None,
              'wall_seconds': time.perf_counter()-started}
    write(args.output/'report.json', report)
    print(report['status'], f'{len(cases)} cases / {len(rows)} logits vectors;', report['max_errors'])
    print('Report:', args.output/'report.json')
    return int(bool(failed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze','check'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, default=ROOT/'build-sharegpt')
    p.add_argument('--baseline', type=Path, default=ROOT/'build-sharegpt-reference')
    p.add_argument('--output', type=Path)
    p.add_argument('--quick', action='store_true', help='Six fixed ShareGPT histories; development only')
    args = p.parse_args()
    for name in ('model','build_dir','baseline'):
        setattr(args, name, getattr(args,name).resolve())
    if args.command == 'freeze':
        if args.quick:
            p.error('Freeze always captures all 100 cases')
        freeze(args)
        return 0
    if args.output is None:
        p.error('check requires --output (an empty result directory)')
    args.output = args.output.resolve()
    return check(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('FAIL:', exc, file=sys.stderr)
        raise SystemExit(1)
