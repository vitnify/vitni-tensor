"""Summarize every frozen test case, including actual failed decodes."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUNS = OUT/'runs'


def read(name):
    path = RUNS/f'{name}.json'
    return json.loads(path.read_text()) if path.exists() else None


def traces(metrics):
    return json.loads(Path(metrics['trace_path']).read_text())


def compare(left, right):
    a, b = traces(left), traces(right)
    result = dict(left_rows=len(a), right_rows=len(b))
    for key in ['logits_sha256', 'cdf_sha256']:
        bad = [i for i, (x, y) in enumerate(zip(a, b)) if x[key] != y[key]]
        result[key] = dict(different_rows=len(bad)+abs(len(a)-len(b)),
                           first_difference=bad[0] if bad else None)
    return result


def main():
    summary = dict(cases=[], completed_runs=0)
    completed = list(RUNS.glob('*.json'))
    summary['completed_runs'] = sum(not p.name.endswith('.trace.json') for p in completed)
    for provider in ['vitni', 'hf', 'gguf']:
        gpu = 'metal' if provider == 'vitni' else 'mps'
        for fixture in ['prose_full', 'code_prefix', 'norwegian_prefix']:
            raw = (OUT/'fixtures'/f'{fixture}.txt').read_bytes()
            case = dict(provider=provider, fixture=fixture, input_bytes=len(raw),
                        input_sha256=hashlib.sha256(raw).hexdigest(), encodes={}, decodes=[])
            for encoding_device in ['cpu', gpu]:
                stem = f'{provider}-{fixture}-from-{encoding_device}'
                enc = read(stem+'-encode')
                if enc is None:
                    continue
                case['encodes'][encoding_device] = enc
                for decoding_device in ['cpu', gpu]:
                    dec = read(stem+'-to-'+decoding_device)
                    if dec is None:
                        continue
                    case['decodes'].append(dict(encoding_device=encoding_device,
                                               decoding_device=decoding_device,
                                               metrics=dec,
                                               encoder_decoder_trace=compare(enc, dec)))
            if 'cpu' in case['encodes'] and gpu in case['encodes']:
                a, b = case['encodes']['cpu'], case['encodes'][gpu]
                case['cpu_gpu_encoding_trace'] = compare(a, b)
                case['identical_archive'] = a['archive_sha256'] == b['archive_sha256']
                case['identical_payload'] = a['payload_sha256'] == b['payload_sha256']
            summary['cases'].append(case)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Completed cold runs:', summary['completed_runs'])
    for c in summary['cases']:
        if not c['encodes']:
            continue
        counts = {'same': [0, 0], 'cross': [0, 0]}
        for d in c['decodes']:
            key = 'same' if d['encoding_device'] == d['decoding_device'] else 'cross'
            counts[key][1] += 1
            counts[key][0] += int(d['metrics'].get('parent_exact_recovery', False))
        sizes = {k: v.get('archive_bytes') for k, v in c['encodes'].items()}
        work = {k: round(v.get('work_seconds', 0), 3) for k, v in c['encodes'].items()}
        print(c['provider'], c['fixture'], 'bytes', sizes, 'encode seconds', work,
              'recovered', counts, 'identical archive', c.get('identical_archive'))


if __name__ == '__main__':
    main()
