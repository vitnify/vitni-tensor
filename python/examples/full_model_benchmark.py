"""Matched resident Rust / eager PyTorch full-engine timings; no kernel substitutions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
from vitni_torch import VitniModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gguf',required=True)
    parser.add_argument('--worker',required=True)
    parser.add_argument('--backend',choices=['cpu','metal'],required=True)
    parser.add_argument('--tokens-json',required=True,help='JSON list of integer token IDs')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--repetitions',type=int,default=9)
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--n-new',type=int,default=32)
    args=parser.parse_args()
    if args.repetitions<3 or args.warmups<1:parser.error('need >=3 repetitions and >=1 warmup')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    tokens=json.loads(Path(args.tokens_json).read_text())
    full=torch.tensor(tokens,dtype=torch.int64)
    prompt=full[:4].clone()
    optimized=args.backend=='metal'
    model_id='torch-full-model-research'
    command=[str(Path(args.worker).resolve()),'--gguf',str(Path(args.gguf).resolve()),
             '--backend',args.backend,'--optimized',str(int(optimized)),
             '--output',str((out/'standalone-last-logits.bin').resolve()),'--model-id',model_id]
    error_log=(out/'native-stderr.log').open('w')
    started=time.perf_counter()
    worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error_log,text=True,bufsize=1)
    model=None
    def exchange(line):
        worker.stdin.write(line+'\n');worker.stdin.flush()
        response=worker.stdout.readline()
        if not response:raise RuntimeError('Rust worker exited; see native-stderr.log')
        return json.loads(response)
    try:
        ready=json.loads(worker.stdout.readline())
        process_load_seconds=time.perf_counter()-started
        started=time.perf_counter()
        model=VitniModel(args.gguf,backend=args.backend,optimized=optimized,model_id=model_id)
        constructor_seconds=time.perf_counter()-started
        result={'backend':args.backend,'torch':torch.__version__,'machine':platform.machine(),
                'scope':'Same resident engine, fresh-sequence reset for every call. Rust timers exclude IPC/file IO and preallocated full-logit output. PyTorch wall includes module call, validation, output allocation, FFI and result conversion. GPU work completes synchronously. No Torch import or model load in warm timings.',
                'standalone_ready':ready,'standalone_process_and_load_seconds':process_load_seconds,
                'pytorch_constructor_seconds':constructor_seconds,'warmups':args.warmups,'repetitions':args.repetitions,
                'startup_scope':'Observed startup with shared filesystem cache; neither cold disk nor fresh Torch import comparison.',
                'cases':[]}
        for mode,tensor in [('forward',full),('generate',prompt)]:
            line=('F ' if mode=='forward' else f'G {args.n_new} ')+','.join(map(str,tensor.tolist()))
            def python_call():
                return model(tensor) if mode=='forward' else model.generate(tensor,n_new=args.n_new)
            first={}
            for variant in ['standalone','pytorch']:
                start=time.perf_counter()
                value=exchange(line)if variant=='standalone'else python_call()
                first[variant]={'wall_seconds':time.perf_counter()-start,
                                'native_seconds':value['seconds']if variant=='standalone'else model.last_native_seconds}
            for _ in range(args.warmups):exchange(line);python_call()
            samples=[]
            for rep in range(args.repetitions):
                observation={'rep':rep,'load_before':os.getloadavg()}
                for variant in (['standalone','pytorch']if rep%2==0 else ['pytorch','standalone']):
                    if variant=='standalone':
                        native=exchange(line);observation['standalone_seconds']=native['seconds']
                    else:
                        start=time.perf_counter();wrapped=python_call()
                        observation['pytorch_wall_seconds']=time.perf_counter()-start
                        observation['pytorch_native_seconds']=model.last_native_seconds
                if mode=='forward':
                    expected=(out/'standalone-last-logits.bin').read_bytes()
                    actual=wrapped.numpy().tobytes()
                    assert actual==expected,'full-logit bit mismatch'
                    observation['output_sha256']=hashlib.sha256(actual).hexdigest()
                else:
                    assert wrapped==native['receipt'],'complete receipt mismatch'
                    observation['model_digest']=wrapped['model_digest']
                samples.append(observation)
                print(mode,rep,'exact',observation['pytorch_wall_seconds']/observation['standalone_seconds'],flush=True)
            medians={key:statistics.median(s[key]for s in samples)for key in
                     ['standalone_seconds','pytorch_wall_seconds','pytorch_native_seconds']}
            result['cases'].append({'mode':mode,'input_tokens':tensor.tolist(),'n_new':args.n_new if mode=='generate' else 0,
                                    'first_call':first,'samples':samples,'median_seconds':medians,
                                    'pytorch_over_standalone':medians['pytorch_wall_seconds']/medians['standalone_seconds'],
                                    'native_over_standalone':medians['pytorch_native_seconds']/medians['standalone_seconds'],
                                    'median_python_boundary_seconds':statistics.median(s['pytorch_wall_seconds']-s['pytorch_native_seconds']for s in samples),
                                    'all_outputs_exact':True})
        result['pytorch_final_info']=model.info()
        result['standalone_final_info']=exchange('INFO')
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if model is not None:model.close()
        if worker.poll()is None:
            worker.stdin.write('QUIT\n');worker.stdin.flush()
            worker.wait(timeout=30)
        error_log.close()

if __name__=='__main__':main()
