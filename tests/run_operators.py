"""Production CUDA operators versus CPU float64 references on quantized BF16 inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch

LENGTHS = [1,2,31,32,33,127,128,129,511,512,513,1023,1024,1025,2047,2048,4096,8192]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--model',type=Path,help='Unused; compatible with the unified runner')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    rng=np.random.default_rng(20260920)
    rows=[]
    def quant(x):
        return torch.from_numpy(np.asarray(x,dtype=np.float32)).to(torch.bfloat16).float().numpy()
    def random(shape,scale=1):
        return quant(rng.standard_normal(shape,dtype=np.float32)*scale)
    def double(x): return torch.from_numpy(x).double()
    def call(op,arrays,a=0,b=0,c=0):
        src=args.output/'operator-fixture.f32'; dst=args.output/'operator-output.f32'
        with src.open('wb') as f:
            for x in arrays: np.asarray(x,dtype=np.float32).tofile(f)
        r=subprocess.run([str(args.probe.resolve()),op,str(src),str(dst),str(a),str(b),str(c),'1'],
                         capture_output=True,text=True,timeout=180)
        if r.returncode: raise RuntimeError(f'{op}: {r.stderr}')
        return np.fromfile(dst,dtype=np.float32)
    def check(name,actual,expected,atol=2e-4,rtol=1/128):
        expected=np.asarray(expected,dtype=np.float64).reshape(-1); actual=actual.reshape(-1)
        if actual.size!=expected.size: raise RuntimeError(f'{name}: output shape mismatch')
        error=np.abs(actual-expected)
        bad=~np.isfinite(actual) | ~np.isfinite(expected) | (error>atol+rtol*np.abs(expected))
        row={'name':name,'elements':actual.size,'atol':atol,'rtol':rtol,'bad_elements':int(bad.sum()),
             'max_absolute_error':float(error.max()),'pass':not bool(bad.any())}
        if bad.any():
            i=int(np.flatnonzero(bad)[0]); row['first_failure']={'index':i,'actual':float(actual[i]),'reference':float(expected[i])}
        rows.append(row)
        print(name,'PASS' if row['pass'] else 'FAIL',flush=True)
        (args.output/'operators-report.json').write_text(json.dumps({'checks':rows,'failures':sum(not r['pass'] for r in rows)},indent=2)+'\n')

    for scale in [0,1e-4,1,20]:
        x=random((1024,),scale); w=random((1024,))
        ref=double(x)/torch.sqrt(double(x).square().mean()+1e-6)*double(w)
        for inplace in [0,1]: check(f'rmsnorm/{scale}/inplace{inplace}',call('rmsnorm',[x,w],c=inplace),ref)
        q=random((16,128),scale); k=random((8,128),scale); qw=random((128,));kw=random((128,))
        ref=torch.cat([(double(q)/torch.sqrt(double(q).square().mean(-1,keepdim=True)+1e-6)*double(qw)).flatten(),
                       (double(k)/torch.sqrt(double(k).square().mean(-1,keepdim=True)+1e-6)*double(kw)).flatten()])
        check(f'qknorm/{scale}',call('qknorm',[q,k,qw,kw]),ref)
        gate=random((3072,),scale);up=random((3072,))
        check(f'swiglu/{scale}',call('swiglu',[gate,up]),torch.nn.functional.silu(double(gate))*double(up))
    for pos in [0,1,127,1023,1024,4096,8191]:
        q=random((16,128));k=random((8,128))
        angles=pos*torch.pow(1000000.,-torch.arange(64,dtype=torch.float64)*2/128)
        def rope(x):
            x=double(x);a,b=x[:,:64],x[:,64:]
            return torch.cat([a*angles.cos()-b*angles.sin(),a*angles.sin()+b*angles.cos()],-1).flatten()
        check(f'rope/{pos}',call('rope',[q,k],a=pos),torch.cat([rope(q),rope(k)]),atol=2e-3)
    x=random((151936,),20);check('convert/exact',call('convert',[x]),x,atol=0,rtol=0)

    for length in LENGTHS:
        for kind in ['uniform','random','peaked','negative']:
            x=np.full((16,8192),-1234,dtype=np.float32)
            active=rng.standard_normal((16,length),dtype=np.float32)*3
            if kind=='uniform': active[:]=0
            if kind=='peaked': active[:]=-1000;active[:,length//2]=1000
            if kind=='negative': active-=1000
            x[:,:length]=active
            actual=call('softmax',[x],a=length).reshape(16,8192)
            check(f'softmax/{length}/{kind}',actual[:,:length],torch.softmax(double(active),-1),atol=2e-6,rtol=2e-5)
            if length<8192: check(f'softmax-tail/{length}/{kind}',actual[:,length:],x[:,length:],atol=0,rtol=0)

    for length,kind in [(n,'random') for n in LENGTHS]+[(n,k) for n in [1,1025,8192] for k in ['zero','peaked']]:
        q=random((16,128),0 if kind=='zero' else (8 if kind=='peaked' else 1))
        # Future cache entries are nonzero and distinct: accidental unmasking is observable.
        k=random((8192,8,128));v=random((8192,8,128))
        scores=torch.einsum('hd,thd->ht',double(q),double(k[:length]).repeat_interleave(2,dim=1))/(128**.5)
        probs=torch.softmax(scores,-1)
        ref=torch.einsum('ht,thd->hd',probs,double(v[:length]).repeat_interleave(2,dim=1))
        out=call('attention',[q,k,v],a=length)
        size=16*8192; native_scores=out[:size].reshape(16,8192);native_probs=out[size:2*size].reshape(16,8192)
        check(f'attention-scores/{length}/{kind}',native_scores[:,:length],scores,atol=3e-5,rtol=3e-5)
        check(f'attention-probs/{length}/{kind}',native_probs[:,:length],probs,atol=3e-6,rtol=3e-5)
        check(f'attention-value/{length}/{kind}',out[2*size:2*size+2048],ref)
        check(f'attention-dispatch/{length}/{kind}',out[2*size+2048:],ref)
        if length<8192:
            check(f'attention-tail/{length}/{kind}',native_probs[:,length:],np.full((16,8192-length),-1234),atol=0,rtol=0)

    for m,n,beta in [(2048,1024,0),(1024,1024,0),(1024,2048,1),
                     (3072,1024,0),(1024,3072,1),(151936,1024,0)]:
        w=random((m,n),1/(n**.5));x=random((n,));y=random((m,))
        # Chunk the FP64 reference to bound CPU memory for the vocabulary projection.
        ref=np.empty(m,dtype=np.float64)
        for start in range(0,m,4096):
            ref[start:start+4096]=(double(w[start:start+4096])@double(x)+beta*double(y[start:start+4096])).numpy()
        check(f'matmul/{m}x{n}/beta{beta}',call('matmul',[w,x,y],a=m,b=n,c=beta),ref)
    report={'seed':20260920,'reference':'CPU float64 on BF16-quantized inputs',
            'probe_sha256':hashlib.sha256(args.probe.read_bytes()).hexdigest(),
            'checks':rows,'failures':sum(not r['pass'] for r in rows),
            'scope':'isolated operator tolerances; not whole-model numerical acceptance'}
    (args.output/'operators-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ['operator-fixture.f32','operator-output.f32']: (args.output/name).unlink()
    print(f'OPERATORS {len(rows)-report["failures"]}/{len(rows)} PASS',flush=True)
    return int(report['failures']>0)


if __name__=='__main__': raise SystemExit(main())
