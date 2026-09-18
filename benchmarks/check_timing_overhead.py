"""Paired ITL instrumentation calibration, independent of optimization experiments."""
import argparse, json, statistics, subprocess, sys, hashlib
from pathlib import Path
from datetime import datetime,timezone
from run_benchmark import sha
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-raw',type=Path,help='Optional trusted generation IDs from a prior identical workload')
    args=p.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be new/empty')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'started':datetime.now(timezone.utc).isoformat(),'order':'ABBA/BAAB',
        'A':'no-itl','B':'full','dataset':'sharegpt100 quick six fixed cases','warmup':1,'repeats':3,
        'primary':'mean of per-case total_ms medians across all 12 samples/mode/case',
        'classification':'Resolved increase only if all 4 adjacent paired changes are positive and the smallest exceeds both within-mode process mean spreads. Otherwise unresolved; this is an engineering screen, not confidence interval or an upper bound.',
        'probe_sha256':sha(args.build_dir/'bin/benchmark_probe'),
        'source_sha256':sha(ROOT.parent/'models/qwen_model.cuh'),
        'tool_sha256':{n:sha(ROOT/n) for n in ['benchmark_probe.cu','run_benchmark.py','metrics.py','check_timing_overhead.py']},
        'reference_raw_sha256':sha(args.reference_raw) if args.reference_raw else None}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    previous={}
    if args.reference_raw:
        for line in args.reference_raw.read_text().splitlines():
            r=json.loads(line)
            if r.get('event')=='request' and not r['warmup']:
                if r['case'] in previous:assert previous[r['case']]==r['generated_ids']
                previous[r['case']]=r['generated_ids']
    all_rows={'A':[],'B':[]};processes=[]
    for i,mode in enumerate('ABBABAAB'):
        dest=out/f'{i}-{mode}'
        cmd=[sys.executable,str(ROOT/'run_benchmark.py'),'--model',str(args.model),
             '--build-dir',str(args.build_dir),'--output',str(dest),'--dataset','sharegpt100',
             '--quick','--warmup','1','--repeats','3']
        if mode=='A':cmd.append('--no-itl')
        print(f'Process {i+1}/8: {protocol[mode]}',flush=True)
        (out/f'{i}-{mode}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (out/f'{i}-{mode}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=read(dest/'summary.json')
        assert summary['probe_sha256']==protocol['probe_sha256']
        assert not summary['memory']['enabled'] and summary['memory']['samples']==0
        assert not (dest/'memory.jsonl').exists()
        rows=[json.loads(s) for s in (dest/'raw.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r.get('event')=='request' and not r['warmup']]
        assert len(rows)==18
        if previous:
            for r in rows:assert r['generated_ids']==previous[r['case']], 'Generation changed '+r['case']
        all_rows[mode].extend(rows)
        processes.append({'index':i,'mode':mode,'mean_total_ms':statistics.mean(r['total_ms'] for r in rows),
                          'gpu_before':summary['gpu_before'],'gpu_after':summary['gpu_after']})
    pairs=[]
    for i in range(0,8,2):
        pair={r['mode']:r['mean_total_ms'] for r in processes[i:i+2]}
        pairs.append((pair['B']/pair['A']-1)*100)
    cases=[]
    for case in dict.fromkeys(r['case'] for r in all_rows['A']):
        rows={m:[r for r in all_rows[m] if r['case']==case] for m in 'AB'}
        assert len(rows['A'])==len(rows['B'])==12
        ids=[r['generated_ids'] for m in 'AB' for r in rows[m]]
        assert all(x==ids[0] for x in ids)
        a,b=[statistics.median(r['total_ms'] for r in rows[m]) for m in 'AB']
        cases.append({'case':case,'prompt_tokens':rows['A'][0]['prompt_tokens'],'output_tokens':rows['A'][0]['output_tokens'],
            'no_itl_ms':a,'full_ms':b,'difference_ms':b-a,'change_percent':(b/a-1)*100,'generation_equal':True})
    a,b=[statistics.mean(c[k] for c in cases) for k in ['no_itl_ms','full_ms']]
    spreads={}
    for m in 'AB':
        values=[r['mean_total_ms'] for r in processes if r['mode']==m]
        spreads[m]=(max(values)-min(values))/statistics.mean(values)*100
    result={'status':'VALIDATION_PASS','timing_protocol':4,'measured_requests':144,'generation_equal':True,
        'reference_generation_equal':True if previous else None,'primary_no_itl_ms':a,'primary_full_ms':b,
        'observed_change_percent':(b/a-1)*100,'paired_changes_percent':pairs,'process_spreads_percent':spreads,
        'interpretation':'RESOLVED_INCREASE' if min(pairs)>max(spreads.values()) else 'NOT_RESOLVED_AT_THIS_REPEAT_COUNT',
        'scope':'Same binary and inference, with/without interior ITL reads/stores. Boundary clocks and common recording costs remain in both modes. Do not subtract this observed difference from timings.',
        'cases':cases,'processes':processes,'finished':datetime.now(timezone.utc).isoformat()}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','processes']},indent=2))
if __name__=='__main__':main()
