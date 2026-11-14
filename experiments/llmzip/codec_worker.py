"""One cold-process run of the pinned, unchanged independent llmzip codec.

Decode receives only an archive and model paths. Source comparisons happen in
the parent, after this process exits. Instrumentation observes and returns the
original CDF; no model results are cached across runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

PROCESS_START = time.perf_counter()
from adapters import up, predictor, PROFILE_IDS
import numpy as np
import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['encode', 'decode'])
    parser.add_argument('provider', choices=['hf', 'gguf', 'vitni'])
    parser.add_argument('device', choices=['cpu', 'mps', 'metal'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('metrics', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(10)
    metrics = dict(operation=args.operation, provider=args.provider,
                   device=args.device, input=str(args.input),
                   output=str(args.output), platform=platform.platform(),
                   torch=torch.__version__, numpy=np.__version__,
                   threads=torch.get_num_threads(), success=False)
    pred = None
    trace = []
    original_cumfreq = up.DistributionModel.cumfreq

    def observed_cumfreq(self, logits):
        cum = original_cumfreq(self, logits)
        trace.append(dict(position=len(trace),
                          logits_sha256=sha(np.asarray(logits, dtype='<f4').tobytes()),
                          cdf_sha256=sha(np.asarray(cum, dtype='<i8').tobytes())))
        return cum

    up.DistributionModel.cumfreq = observed_cumfreq
    try:
        raw = args.input.read_bytes()
        metrics['input_sha256'] = sha(raw)
        if args.operation == 'decode':
            hdr, offset = up.unpack_header(raw)
            if hdr['mode'] != up.MODE_ARITH:
                raise ValueError('arithmetic model mode required; raw fallback rejected')
            if hdr['model_id'] != PROFILE_IDS[args.provider]:
                raise ValueError('archive numerical profile/model hash does not match selected backend')
            window, keep = hdr['window'], hdr['keep']
        else:
            window, keep = 256, 64
        pred = predictor(args.provider, args.device, window, keep,
                         log=args.metrics.with_suffix('.native.stderr'))
        metrics.update(load_seconds=pred.load_seconds, window=window, keep=keep,
                       numerical_profile=pred.model_id)
        if args.provider == 'vitni':
            metrics['native_ready'] = pred.ready
        start = time.perf_counter()
        try:
            if args.operation == 'encode':
                blob, stats = up.compress_text(raw.decode('utf-8'), pred,
                                               mode=up.MODE_ARITH, quiet=True,
                                               temperature=1., mix_weight=0.)
                hdr, offset = up.unpack_header(blob)
                if hdr['mode'] != up.MODE_ARITH:
                    raise ValueError('raw fallback rejected')
                args.output.write_bytes(blob)
                metrics.update(stats=stats, archive_bytes=len(blob),
                               archive_sha256=sha(blob),
                               payload_sha256=sha(blob[offset:]))
            else:
                # The only text available here is what the original decoder
                # reconstructs. Its original integrity check must pass first.
                text = up.decompress_blob(raw, pred=pred, quiet=True)
                recovered = text.encode('utf-8')
                args.output.write_bytes(recovered)
                metrics.update(recovered_bytes=len(recovered),
                               recovered_sha256=sha(recovered))
            metrics['success'] = True
        finally:
            metrics['work_seconds'] = time.perf_counter() - start
            metrics.update(forward_seconds=pred.forward_seconds,
                           forward_positions=pred.forward_positions,
                           cache_resets=pred.resets)
            if 'hdr' in locals():
                metrics['header'] = {k: v.hex() if isinstance(v, bytes) else v
                                     for k, v in hdr.items()}
    except Exception as exc:
        metrics.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        if pred is not None:
            pred.close()
        trace_path = args.metrics.with_suffix('.trace.json')
        trace_path.write_text(json.dumps(trace, indent=2) + '\n')
        metrics.update(trace_rows=len(trace), trace_path=str(trace_path),
                       process_seconds=time.perf_counter() - PROCESS_START)
        args.metrics.write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps({k: metrics.get(k) for k in
                          ('operation', 'provider', 'device', 'success',
                           'work_seconds', 'archive_bytes', 'trace_rows', 'error')}), flush=True)
    return 0 if metrics['success'] else 2


if __name__ == '__main__':
    sys.exit(main())
