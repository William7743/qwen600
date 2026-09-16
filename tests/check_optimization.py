"""Freeze trusted V0 outputs or apply fixed V0/V1 numerical-equivalence limits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(path.read_text())


def logit_metrics(base,candidate):
    a=base.astype(np.float64);b=candidate.astype(np.float64)
    pa=np.exp(a-a.max());pa/=pa.sum()
    pb=np.exp(b-b.max());pb/=pb.sum()
    return {'mean_absolute_error':float(np.abs(a-b).mean()),
            'max_absolute_error':float(np.abs(a-b).max()),
            'probability_total_variation':float(np.abs(pa-pb).sum()/2),
            'baseline_top1_regret':float(a.max()-a[int(b.argmax())]),
            'baseline_top1':int(a.argmax()),'candidate_top1':int(b.argmax())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--results',type=Path,required=True)
    f.add_argument('--baseline',type=Path,required=True)
    c=sub.add_parser('compare');c.add_argument('--baseline',type=Path,required=True)
    c.add_argument('--candidate',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    args=p.parse_args();policy=read(ROOT/'optimization_policy.json')
    if args.command=='freeze':
        if args.baseline.exists() and any(args.baseline.iterdir()):
            raise RuntimeError('Baseline directory must be empty; refusing to overwrite V0')
        report=read(args.results/'extended-model-report.json')
        operators=read(args.results/'operators-report.json')
        if len(report['cases'])!=33 or report['failures'] or operators['failures'] or len(operators['checks'])!=287:
            raise RuntimeError('Incomplete or failing baseline diagnostics')
        names=['extended-model-report.json','operators-report.json']
        for case in report['cases']:
            for mode in ['forward','trace']:
                names.extend([case['name']+f'.{mode}.native.f32',case['name']+f'.{mode}.ids'])
        args.baseline.mkdir(parents=True,exist_ok=True)
        hashes={}
        for name in names:
            shutil.copyfile(args.results/name,args.baseline/name);hashes[name]=sha(args.baseline/name)
        manifest={'version':1,'policy':policy,'artifacts':hashes,
                  'source_probe_sha256':report['probe_sha256'],
                  'corpus_sha256':report['corpus_sha256'],
                  'scope':'Frozen current implementation; Transformers numerical acceptance remains pending'}
        (args.baseline/'baseline.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Frozen V0 reference:',args.baseline)
        return 0

    manifest=read(args.baseline/'baseline.json')
    if manifest['policy']!=policy: raise RuntimeError('Policy differs from frozen baseline')
    for name,digest in manifest['artifacts'].items():
        if sha(args.baseline/name)!=digest: raise RuntimeError(f'Baseline modified: {name}')
    base=read(args.baseline/'extended-model-report.json')
    cand=read(args.candidate/'extended-model-report.json')
    if cand['corpus_sha256']!=manifest['corpus_sha256']:
        raise RuntimeError('Candidate corpus differs from V0')
    if [x['name'] for x in cand['cases']] != [x['name'] for x in base['cases']]:
        raise RuntimeError('Candidate cases incomplete or reordered')
    baseline_ops=read(args.baseline/'operators-report.json')
    candidate_ops=read(args.candidate/'operators-report.json')
    operator_schema=lambda r:[(x['name'],x['elements'],x['atol'],x['rtol']) for x in r['checks']]
    if operator_schema(candidate_ops)!=operator_schema(baseline_ops):
        raise RuntimeError('Operator coverage or tolerances changed')
    failures=[]; rows=[]
    if candidate_ops['failures'] or any(not x['pass'] or x['bad_elements'] for x in candidate_ops['checks']):
        failures.append({'case':'operators','reason':'isolated operator failure'})
    for old,new in zip(base['cases'],cand['cases']):
        name=old['name']
        for mode in ['forward','trace']:
            filename=name+f'.{mode}.ids'
            if sha(args.candidate/filename)!=manifest['artifacts'][filename]:
                raise RuntimeError(f'Different input/history: {filename}')
        positions=[r['position'] for r in old['logits']]
        if positions!=[r['position'] for r in new['logits']]:
            raise RuntimeError(f'{name}: logits positions changed')
        old_trace=[(r['position'],r['stage']) for r in old['hidden_states']]
        if old_trace!=[(r['position'],r['stage']) for r in new['hidden_states']]:
            raise RuntimeError(f'{name}: hidden-state coverage changed')
        case_failures=0
        for mode,width,keys in [('forward',151936,positions),('trace',1024,old_trace)]:
            filename=name+f'.{mode}.native.f32'
            a=np.fromfile(args.baseline/filename,dtype=np.float32)
            b=np.fromfile(args.candidate/filename,dtype=np.float32)
            if a.size!=len(keys)*width or b.size!=a.size:
                raise RuntimeError(f'{filename}: unexpected tensor size')
            a=a.reshape(-1,width);b=b.reshape(-1,width)
            for key,x,y in zip(keys,a,b):
                metrics={};passed=bool(np.isfinite(x).all() and np.isfinite(y).all())
                if passed and mode=='forward':
                    metrics=logit_metrics(x,y)
                    passed=all(metrics[k]<=v for kmax,v in policy['logits'].items() for k in [kmax.removesuffix('_max')])
                elif passed:
                    x=x.astype(np.float64);y=y.astype(np.float64)
                    relative=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30))
                    maximum=float(np.abs(x-y).max())
                    limits=policy['hidden_states']
                    bound=limits['max_absolute_error_atol']+limits['max_absolute_error_scale']*float(np.abs(x).max())
                    metrics={'relative_l2_error':relative,'max_absolute_error':maximum,'max_absolute_error_limit':bound}
                    passed=relative<=limits['relative_l2_error_max'] and maximum<=bound
                    if key[1]==0: passed=passed and np.array_equal(x,y)
                row={'case':name,'kind':mode,'position_stage':key,'pass':passed,**metrics}
                rows.append(row)
                if not passed: failures.append(row);case_failures+=1
        print(name,'PASS' if not case_failures else f'FAIL ({case_failures} snapshots)',flush=True)
    report={'status':'PASS' if not failures else 'FAIL','scope':policy['scope'],
            'policy':policy,'baseline_manifest_sha256':sha(args.baseline/'baseline.json'),
            'checks':len(rows),'failure_count':len(failures),'failures':failures,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('OPTIMIZATION EQUIVALENCE',report['status'],'Report:',args.output)
    return int(bool(failures))


if __name__=='__main__': raise SystemExit(main())
