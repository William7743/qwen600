"""Offline logits-first iteration. Frozen references; no Transformers during check/full/diagnose."""
import argparse
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from check_optimization import sha, read, logit_metrics

ROOT = Path(__file__).resolve().parents[1]
QUICK = ['en_summary', 'zh_summary', 'cpp', 'whitespace', 'length_129', 'length_1025']
BENCH = ['p16_g16', 'p256_g16', 'p1024_g16', 'p16_g256']


def fresh(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output must be empty: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)


def checked(base, manifest, name):
    path = base / name
    if sha(path) != manifest['artifacts'][name]:
        raise ValueError('Reference hash mismatch: ' + name)
    return path


def prepare(args):
    old = read(args.baseline / 'baseline.json')
    policy = read(ROOT / 'tests/optimization_policy.json')['logits']
    if old['policy']['logits'] != policy:
        raise ValueError('Frozen logits limits differ')
    report = read(checked(args.baseline, old, 'extended-model-report.json'))
    if len(report['cases']) != 33 or report['failures']:
        raise ValueError('Incomplete V0 reference')
    fresh(args.output)
    manifest = {'version': 1, 'logits_limits': policy, 'quick_cases': QUICK,
                'model_sha256': read(ROOT/'tests/reference.json')['model']['sha256'],
                'parent_manifest_sha256': sha(args.baseline/'baseline.json'),
                'cases': [], 'artifacts': {}}
    def copy(source, name):
        shutil.copyfile(source, args.output/name)
        manifest['artifacts'][name] = sha(args.output/name)
        return name
    for case in report['cases']:
        name = case['name']
        row = {'name': name, 'prompt_tokens': case['input_tokens'],
               'positions': [r['position'] for r in case['logits']],
               'trace_positions': sorted({r['position'] for r in case['hidden_states']})}
        for mode in ['forward', 'trace']:
            for suffix in ['ids', 'native.f32']:
                fn = name+'.'+mode+'.'+suffix
                row[mode+'_'+suffix] = copy(checked(args.baseline, old, fn), fn)
        manifest['cases'].append(row)
    # One-time import of trusted V0 long-context output. Never compute it with V1.
    long_report = read(args.long_reference/'report.json')
    if long_report['structural_failures'] or long_report['generation_failures']:
        raise ValueError('Trusted long-context run has regression failures')
    case = next(c for c in long_report['forward'] if c['name']=='boundary')
    if case['input_tokens'] != 8192 or len(case['positions']) != 11:
        raise ValueError('Expected the trusted 8192-token V0 boundary run')
    row = {'name': 'boundary_8192', 'prompt_tokens': 8192,
           'positions': [r['position'] for r in case['positions']]}
    for suffix in ['ids', 'logits.f32']:
        source = args.long_reference/('boundary.'+suffix)
        key = 'forward_ids' if suffix=='ids' else 'forward_native.f32'
        row[key] = copy(source, 'boundary_8192.'+suffix)
    if len((args.output/row['forward_ids']).read_text().split()) != 8192:
        raise ValueError('Invalid long history')
    if (args.output/row['forward_native.f32']).stat().st_size != 11*151936*4:
        raise ValueError('Invalid long logits')
    manifest['long_report_sha256'] = sha(args.long_reference/'report.json')
    manifest['cases'].append(row)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PREPARED', args.output, 'manifest SHA256', sha(args.output/'manifest.json'))


def evaluate(args):
    started = time.perf_counter()
    manifest = read(args.baseline/'manifest.json')
    policy = read(ROOT/'tests/optimization_policy.json')['logits']
    if manifest['version'] != 1 or manifest['logits_limits'] != policy or manifest['quick_cases'] != QUICK:
        raise ValueError('Reference version, selection or limits changed')
    for name, digest in manifest['model_sha256'].items():
        if sha(args.model/name) != digest:
            raise ValueError('Model hash mismatch: '+name)
    cases = manifest['cases']
    if len(cases)!=34 or len({c['name'] for c in cases})!=34:
        raise ValueError('Expected complete 34-case reference')
    trace = args.command == 'diagnose'
    if args.command == 'check':
        cases = [next(c for c in cases if c['name']==name) for name in QUICK]
    elif trace:
        cases = [next(c for c in cases if c['name']==args.case)]
        if 'trace_positions' not in cases[0]:
            raise ValueError('Long reference has logits only. Reproduce on a shorter case or capture trusted V0 layer traces at the failing position.')
    probe = args.build_dir.resolve()/'bin/iteration_probe'
    fresh(args.output)
    plan = []
    expected = []
    for c in cases:
        mode = 'trace' if trace else 'forward'
        ids = checked(args.baseline, manifest, c[mode+'_ids']).resolve()
        positions = c['trace_positions'] if trace else c['positions']
        ref = checked(args.baseline, manifest, c[mode+'_native.f32'])
        width = 30*1024 if trace else 151936
        values = np.fromfile(ref, dtype=np.float32)
        if values.size != len(positions)*width or not np.isfinite(values).all():
            raise ValueError('Invalid reference array: '+c['name'])
        expected.append(values.reshape(len(positions), -1))
        plan.append(f'{json.dumps(str(ids),ensure_ascii=False)} {c["prompt_tokens"]} {len(positions)} '+ ' '.join(map(str,positions)))
    (args.output/'plan.txt').write_text('\n'.join(plan)+'\n')
    command = [str(probe), str((args.model/'model.safetensors').resolve()),
               str((args.output/'plan.txt').resolve()), str((args.output/'actual.f32').resolve()),
               'trace' if trace else 'logits']
    if args.sanitizer:
        command = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '1']+command
    native_start = time.perf_counter()
    with (args.output/'native.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=args.timeout_seconds or (7200 if args.sanitizer else 1800))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError('Probe timed out; entire probe/sanitizer process group stopped')
    native_seconds = time.perf_counter()-native_start
    if process.returncode:
        raise RuntimeError('Probe or sanitizer failed; see '+str(args.output/'native.log'))
    rows = []
    if trace:
        dtype = np.dtype([('position','<i4'),('stage','<i4'),('values','<f4',(1024,))])
        data = np.fromfile(args.output/'actual.f32', dtype=dtype)
        c = cases[0]; coordinates = [(p,s) for p in c['trace_positions'] for s in range(30)]
        if (args.output/'actual.f32').stat().st_size != len(coordinates)*dtype.itemsize or [(int(r['position']),int(r['stage'])) for r in data] != coordinates:
            raise ValueError('Missing/reordered layer hooks: preserve layer-output tracing, not internal operator boundaries')
        limits = read(ROOT/'tests/optimization_policy.json')['hidden_states']
        for (pos,stage), ref, row in zip(coordinates,expected[0].reshape(-1,1024),data):
            actual = row['values'].astype(np.float64);ref=ref.astype(np.float64)
            finite = bool(np.isfinite(actual).all())
            rel = float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)) if finite else None
            maximum = float(np.abs(actual-ref).max()) if finite else None
            bound = limits['max_absolute_error_atol']+limits['max_absolute_error_scale']*float(np.abs(ref).max())
            passed = finite and rel<=limits['relative_l2_error_max'] and maximum<=bound and (stage!=0 or np.array_equal(ref,actual))
            rows.append({'case':c['name'],'position':pos,'stage':stage,'pass':passed,'exact':bool(np.array_equal(ref,actual)),'relative_l2':rel,'max_absolute_error':maximum})
            print(f'position={pos} stage={stage}: '+('PASS' if passed else 'FAIL'),flush=True)
    else:
        data = np.fromfile(args.output/'actual.f32', dtype=np.float32)
        size = sum(x.size for x in expected)
        if data.size != size or (args.output/'actual.f32').stat().st_size != size*4:
            raise ValueError(f'Missing or extra logits: expected {size}, got {data.size}')
        offset = 0
        for c, ref in zip(cases, expected):
            actual = data[offset:offset+ref.size].reshape(ref.shape);offset+=ref.size
            failures = 0
            for pos,x,y in zip(c['positions'],ref,actual):
                finite = bool(np.isfinite(y).all())
                metrics = logit_metrics(x,y) if finite else {}
                passed = finite and all(metrics[k.removesuffix('_max')]<=v for k,v in policy.items())
                rows.append({'case':c['name'],'position':pos,'pass':passed,**metrics})
                if not passed:
                    failures+=1;print('FAIL',c['name'],'position',pos,metrics,flush=True)
            print(c['name'], 'PASS' if not failures else 'FAIL',flush=True)
    failed = [r for r in rows if not r['pass']]
    status = 'FAIL' if failed else ('DIAGNOSTIC_PASS' if trace else 'QUICK_LOGITS_PASS' if args.command=='check' else 'FULL_LOGITS_PASS')
    report = {'status':status,'scope':'V0 equivalence at sampled boundaries; not standalone release or HF certification',
              'baseline_manifest_sha256':sha(args.baseline/'manifest.json'),'probe_sha256':sha(probe),
              'logits_limits':policy,'checks':len(rows),'failures':failed,'results':rows,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'sanitizer':args.sanitizer,'native_seconds':native_seconds,'wall_seconds':time.perf_counter()-started}
    if trace:
        report['first_nonexact_stage']={str(pos):next((r['stage'] for r in rows if r['position']==pos and not r['exact']),None) for pos in cases[0]['trace_positions']}
        report['first_failing_stage']={str(pos):next((r['stage'] for r in rows if r['position']==pos and not r['pass']),None) for pos in cases[0]['trace_positions']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(status, f'{report["wall_seconds"]:.2f}s total; {native_seconds:.2f}s native. Report:',args.output/'report.json')
    if failed and not trace:
        print('Next: iterate.py diagnose --case',failed[0]['case'],'with the same baseline/model/build arguments')
    return int(bool(failed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','check','full','diagnose','bench'])
    p.add_argument('--model',type=Path,default=Path('/home/msganzy/vllm-shared/models/Qwen3-0.6B'))
    p.add_argument('--baseline',type=Path,default=ROOT/'build-iteration-reference')
    p.add_argument('--long-reference',type=Path)
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-iteration')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case')
    p.add_argument('--sanitizer',choices=['memcheck','racecheck'])
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--timeout-seconds',type=int,help='Native timeout; default 1800, or 7200 with sanitizer')
    p.add_argument('--long-decode',action='store_true',help='Add 1024/256 to quick benchmark')
    args=p.parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds<=0:p.error('timeout must be positive')
    if args.command=='prepare':
        if not args.long_reference:p.error('prepare requires --long-reference from trusted V0')
        prepare(args);return 0
    if args.command=='bench':
        command=[sys.executable,str(ROOT/'benchmarks/run_benchmark.py'),'--model',str(args.model),
                 '--build-dir',str(args.build_dir),'--output',str(args.output),'--dataset','fixed9',
                 '--warmup','1','--repeats',str(args.repeats)]
        selected=BENCH+(['p1024_g256'] if args.long_decode else [])
        for case in selected:command+=['--case',case]
        print('QUICK benchmark: selected cases, not the full benchmark score',flush=True)
        start=time.perf_counter()
        result=subprocess.run(command)
        elapsed=time.perf_counter()-start
        if result.returncode==0:
            (args.output/'iteration-profile.json').write_text(json.dumps({'profile':'quick','cases':selected,'wall_seconds':elapsed,'repeats':args.repeats},indent=2)+'\n')
        print(f'Quick benchmark wall time: {elapsed:.2f}s')
        return result.returncode
    if args.command=='diagnose' and not args.case:p.error('diagnose requires --case')
    try:
        return evaluate(args)
    except Exception as e:
        print('ERROR:',e,file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
