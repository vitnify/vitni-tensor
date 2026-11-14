"""Short real-codec trial: fresh CPU/Metal encode/decode, exact file recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import run_matrix as runner
from analyze import compare

OUT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=OUT/'fixtures/code_prefix.txt',
                        help='UTF-8 file; default is the pinned 512-byte upstream code sample')
    args = parser.parse_args()
    source = args.input.resolve()
    original = source.read_bytes()
    quick = OUT/'runs-quick'
    quick.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='trial-', dir=quick))
    runner.RUNS = run_dir
    encodes = {}
    decodes = []
    for encoder in ['cpu', 'metal']:
        archive = run_dir/f'from-{encoder}.llmz'
        enc = runner.run(f'from-{encoder}-encode', 'encode', 'vitni', encoder, source, archive)
        if not enc['success']:
            raise RuntimeError(f'encoding failed; see {run_dir}')
        encodes[encoder] = enc
        for decoder in ['cpu', 'metal']:
            target = run_dir/f'from-{encoder}-to-{decoder}.recovered'
            dec = runner.run(f'from-{encoder}-to-{decoder}', 'decode', 'vitni', decoder, archive, target)
            if not dec['success'] or target.read_bytes() != original:
                raise RuntimeError(f'exact recovery failed; see {run_dir}')
            comparison = compare(enc, dec)
            if any(comparison[k]['different_rows'] for k in ['logits_sha256', 'cdf_sha256']):
                raise RuntimeError(f'logit/CDF replay differs; see {run_dir}')
            decodes.append(dict(encoder=encoder, decoder=decoder, exact=True,
                                work_seconds=dec['work_seconds']))
    cpu = Path(encodes['cpu']['output']).read_bytes()
    metal = Path(encodes['metal']['output']).read_bytes()
    if cpu != metal:
        raise RuntimeError('independent CPU/Metal encoders produced different archive bytes')
    summary = dict(success=True, input_bytes=len(original),
                   input_sha256=hashlib.sha256(original).hexdigest(),
                   archive_bytes=len(cpu), identical_archives=True,
                   encodes=encodes, decodes=decodes,
                   scope='one Mac CPU/Metal pairing; pinned experimental numerical profile')
    (run_dir/'trial-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'PASS: all four fresh decoders recovered {len(original)} exact bytes; '
          f'CPU/Metal archives are identical ({len(cpu)} bytes).')
    print('Evidence:', run_dir)


if __name__ == '__main__':
    main()
