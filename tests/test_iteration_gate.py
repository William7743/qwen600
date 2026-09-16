"""Fault-injection checks for the gate itself; fake probes never claim GPU correctness."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    script=Path(__file__).with_name('iterate.py').resolve()
    with tempfile.TemporaryDirectory(prefix='qwen-gate-test-') as directory:
        temp=Path(directory);(temp/'bin').mkdir()
        probe=temp/'bin/iteration_probe'
        probe.write_text('#!'+sys.executable+'\n'+'''import json,os,shlex,sys,time
from pathlib import Path
import numpy as np
base=Path(os.environ['TEST_BASELINE'])
m=json.loads((base/'manifest.json').read_text())
byfile={str((base/c['forward_ids']).resolve()):c for c in m['cases']}
arrays=[]
for line in Path(sys.argv[2]).read_text().splitlines():
 c=byfile[shlex.split(line)[0]]
 arrays.append(np.fromfile(base/c['forward_native.f32'],np.float32))
a=np.concatenate(arrays)
mode=os.environ['TEST_FAULT']
if mode=='timeout':time.sleep(10)
if mode=='within_limit': a[0]+=0.01
if mode=='over_limit': a[0]+=1.0
if mode=='nan': a[0]=float('nan')
if mode=='truncated': a=a[:-1]
if mode=='extra': a=np.append(a,np.float32(0))
a.astype(np.float32).tofile(sys.argv[3])
if mode=='extra_byte':
 with open(sys.argv[3],'ab') as f:f.write(b'x')
''')
        probe.chmod(0o755)
        for fault,expected in [('none',0),('within_limit',0),('over_limit',1),('nan',1),('truncated',2),('extra',2),('extra_byte',2),('timeout',2)]:
            command=[sys.executable,str(script),'check','--baseline',str(args.baseline.resolve()),
                     '--model',str(args.model.resolve()),'--build-dir',str(temp),'--output',str(temp/fault)]
            if fault=='timeout':command+=['--timeout-seconds','1']
            result=subprocess.run(command,capture_output=True,text=True,
                                  env=dict(os.environ,TEST_BASELINE=str(args.baseline.resolve()),TEST_FAULT=fault),timeout=60)
            assert result.returncode==expected,(fault,result.stdout,result.stderr)
            print(fault,'PASS',flush=True)
        # Tamper only with an isolated symlink overlay, never the frozen reference.
        bad=temp/'bad-reference';bad.mkdir()
        for f in args.baseline.iterdir(): (bad/f.name).symlink_to(f.resolve())
        c=json.loads((bad/'manifest.json').read_text())['cases'][0]
        ids=bad/c['forward_ids'];original=ids.read_text();ids.unlink();ids.write_text(original+'\n1\n')
        result=subprocess.run([sys.executable,str(script),'check','--baseline',str(bad),
                               '--model',str(args.model.resolve()),'--build-dir',str(temp),'--output',str(temp/'tampered')],
                              capture_output=True,text=True,timeout=60)
        assert result.returncode==2 and 'hash mismatch' in result.stderr,result.stderr
        print('tampered_reference PASS')


if __name__=='__main__':main()
