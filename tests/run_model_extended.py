"""Broader model diagnostics with per-layer hidden states; numeric acceptance remains pending."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_correctness import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
                dtype=torch.bfloat16,attn_implementation='eager').eval().cuda()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20260920)
    cases=[]
    for item in json.loads((root/'model_cases.json').read_text()):
        messages=([{'role':'system','content':item['system']}] if 'system' in item else [])
        messages.append({'role':'user','content':item['text']})
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                          enable_thinking=item.get('thinking',False))
        if not isinstance(ids,list): ids=ids['input_ids']
        cases.append((item['name'],ids,True))
    seed=tokenizer.encode('A cache stores earlier keys and values. 缓存保存之前的信息。\n',add_special_tokens=False)
    for n in [127,128,129,511,512,513,1023,1024,1025]:
        cases.append((f'length_{n}',(seed*(n//len(seed)+1))[:n],False))
    captured=[]; capture_enabled=False
    def hook(module,inputs,output):
        if capture_enabled:
            value=output[0] if isinstance(output,tuple) else output
            captured.append(value[0,-1].detach().float().cpu().numpy().copy())
    handles=[layer.register_forward_hook(hook) for layer in model.model.layers]
    report={'scope':'33 cases, fixed snapshots plus eight teacher-forced generation steps for 24 texts',
            'reference':'same local weights; Transformers BF16 eager, one token per step',
            'numerical_acceptance':'PENDING','corpus_sha256':hashlib.sha256((root/'model_cases.json').read_bytes()).hexdigest(),
            'probe_sha256':hashlib.sha256(args.probe.read_bytes()).hexdigest(),'cases':[],'failures':0}
    def save(): (args.output/'extended-model-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def native(mode,ids,positions,name):
        src=args.output/(name+'.'+mode+'.ids');src.write_text(''.join(f'{i}\n' for i in ids))
        dst=args.output/(name+'.'+mode+'.native.f32')
        subprocess.run([str(args.probe.resolve()),mode,str(args.model.resolve()),str(src),str(dst),
                        ' '.join(map(str,positions))],check=True,capture_output=True,text=True,timeout=300)
        return np.fromfile(dst,dtype=np.float32)
    with torch.inference_mode():
        for name,ids,generate in cases:
            selected={0,len(ids)//2,len(ids)-1}
            selected.update(n for n in [31,127,511,1023,1024] if n<len(ids))
            trace_positions=sorted({0,len(ids)-1})
            references={};traces={};cache=None;captured=[]
            def step(token,pos):
                nonlocal cache,capture_enabled,captured
                capture_enabled=pos in trace_positions;captured=[]
                out=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,
                          use_cache=True,output_hidden_states=capture_enabled)
                cache=out.past_key_values
                logits=out.logits[0,-1].float().cpu().numpy()
                if pos in selected: references[pos]=logits
                if capture_enabled:
                    assert len(captured)==28
                    traces[pos]=np.stack([out.hidden_states[0][0,-1].float().cpu().numpy(),
                                         *captured,out.hidden_states[-1][0,-1].float().cpu().numpy()])
                return logits
            for pos,token in enumerate(ids): logits=step(token,pos)
            # Use the reference history for both engines; no divergence amplification.
            teacher_ids=list(ids); targets=[]
            if generate:
                for i in range(8):
                    pos=len(teacher_ids)-1;selected.add(pos);references[pos]=logits
                    targets.append(int(logits.argmax()))
                    if i<7:
                        teacher_ids.append(targets[-1]); logits=step(targets[-1],len(teacher_ids)-1)
            positions=sorted(selected)
            actual=native('forward',teacher_ids,positions,name).reshape(len(positions),151936)
            ref=np.stack([references[i] for i in positions]);ref.astype(np.float32).tofile(args.output/(name+'.logits.reference.f32'))
            logits_rows=[{'position':pos,**metrics(actual[i],ref[i])} for i,pos in enumerate(positions)]
            actual_trace=native('trace',ids,trace_positions,name).reshape(len(trace_positions),30,1024)
            ref_trace=np.stack([traces[i] for i in trace_positions]);ref_trace.tofile(args.output/(name+'.trace.reference.f32'))
            hidden=[]
            for i,pos in enumerate(trace_positions):
                for stage in range(30):
                    a=actual_trace[i,stage].astype(np.float64);b=ref_trace[i,stage].astype(np.float64)
                    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
                    hidden.append({'position':pos,'stage':stage,'finite':finite,
                        'exact':bool(np.array_equal(a,b)), 'mean_absolute_error':float(np.abs(a-b).mean()),
                        'max_absolute_error':float(np.abs(a-b).max()),
                        'relative_l2_error':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))})
            failures=sum(not r['finite'] for r in logits_rows)+sum(not r['finite'] or
                            (r['stage']==0 and not r['exact']) for r in hidden)
            report['failures']+=failures
            report['cases'].append({'name':name,'input_tokens':len(ids),'logits':logits_rows,'hidden_states':hidden,
                'first_nonexact_stage':{str(pos):next((r['stage'] for r in hidden if r['position']==pos and not r['exact']),None)
                                        for pos in trace_positions},
                'teacher_forced_reference_ids':targets,
                'teacher_forced_native_ids':[int(actual[positions.index(len(ids)-1+i)].argmax()) for i in range(len(targets))]})
            save()
            print(f'EXTENDED {name}: {len(positions)} logits positions, {len(hidden)} hidden snapshots; '
                  +('FAIL' if failures else 'FINITE_EMBEDDING_CHECKS_PASS; NUMERICAL_REVIEW_REQUIRED'),flush=True)
    for handle in handles: handle.remove()
    report['status']='FAIL' if report['failures'] else 'DIAGNOSTICS_COMPLETE_NUMERICAL_REVIEW_REQUIRED';save()
    return int(report['failures']>0)


if __name__=='__main__': raise SystemExit(main())
