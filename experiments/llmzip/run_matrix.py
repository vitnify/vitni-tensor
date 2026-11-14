"""Serial, resumable cold-process matrix; preserves negative outcomes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PYTHON = Path(sys.executable)
WORKER = OUT/'codec_worker.py'
RUNS = OUT/'runs'
RUNS.mkdir(exist_ok=True)
ENV = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', PYTORCH_ENABLE_MPS_FALLBACK='0',
           OMP_NUM_THREADS='10', VECLIB_MAXIMUM_THREADS='10')


def run(name, operation, provider, device, source, target):
    metrics_path = RUNS/f'{name}.json'
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    cmd = [str(PYTHON), str(WORKER), operation, provider, device,
           str(source), str(target), str(metrics_path)]
    start = time.perf_counter()
    print('START '+name, flush=True)
    with (RUNS/f'{name}.log').open('w') as log:
        proc = subprocess.run(cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter()-start
    if not metrics_path.exists():
        raise RuntimeError(f'{name}: no metrics, exit {proc.returncode}; inspect log')
    metrics = json.loads(metrics_path.read_text())
    metrics.update(command=cmd, exit_code=proc.returncode, outer_seconds=elapsed)
    metrics_path.write_text(json.dumps(metrics, indent=2)+'\n')
    print('DONE '+name+' '+json.dumps({k: metrics.get(k) for k in
          ('success', 'work_seconds', 'outer_seconds', 'archive_bytes', 'error')}), flush=True)
    return metrics


def main():
    # All fixtures were frozen before any archive result. Full English prose
    # comes first to include repeated sliding-window cache rebuilds.
    selected = sys.argv[1:] or ['vitni', 'hf', 'gguf']
    for provider in selected:
        gpu = 'metal' if provider == 'vitni' else 'mps'
        for fixture in ['prose_full', 'code_prefix', 'norwegian_prefix']:
            source = OUT/'fixtures'/f'{fixture}.txt'
            source_raw = source.read_bytes()
            for encoding_device in ['cpu', gpu]:
                stem = f'{provider}-{fixture}-from-{encoding_device}'
                archive = RUNS/f'{stem}.llmz'
                enc = run(stem+'-encode', 'encode', provider, encoding_device,
                          source, archive)
                if not enc['success']:
                    raise RuntimeError(f'encode failed: {stem}')
                for decoding_device in ['cpu', gpu]:
                    name = stem+'-to-'+decoding_device
                    target = RUNS/f'{name}.recovered'
                    dec = run(name, 'decode', provider, decoding_device, archive, target)
                    # Source is read by the parent only, after child termination.
                    if dec['success']:
                        if target.read_bytes() != source_raw:
                            raise RuntimeError('upstream digest accepted nonidentical bytes')
                        dec['parent_exact_recovery'] = True
                    else:
                        dec['parent_exact_recovery'] = False
                    path = RUNS/f'{name}.json'
                    path.write_text(json.dumps(dec, indent=2)+'\n')
                    if encoding_device == decoding_device and not dec['success']:
                        raise RuntimeError(f'same-backend control failed: {name}')
                    if provider == 'vitni' and not dec['success']:
                        raise RuntimeError(f'Vitni portability failed: {name}')
    print('MATRIX COMPLETE', flush=True)


if __name__ == '__main__':
    main()
