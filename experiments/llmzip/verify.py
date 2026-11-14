"""Check the complete evidence bundle; does not rerun engine computations."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    s = json.loads((OUT/'summary.json').read_text())
    assert s['completed_runs'] == 54, s['completed_runs']
    counts = dict(encodes=0, same_device_passes=0, cross_device_passes={},
                  cross_device_failures={}, vitni_coded_positions=0,
                  vitni_compared_vocabulary_words=0)
    for case in s['cases']:
        provider = case['provider']
        raw = (OUT/'fixtures'/f"{case['fixture']}.txt").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == case['input_sha256']
        assert len(case['encodes']) == 2 and len(case['decodes']) == 4
        for e in case['encodes'].values():
            assert e['success']
            assert e['input_sha256'] == case['input_sha256']
            assert e['header']['mode'] == 0 and e['stats']['mode'] == 0
            assert e['header']['prec_bits'] == 30
            assert e['header']['window'] == 256 and e['header']['keep'] == 64
            assert e['trace_rows'] == e['stats']['n_tokens']
            counts['encodes'] += 1
        for d in case['decodes']:
            m = d['metrics']
            if m['success']:
                assert m['parent_exact_recovery']
                assert Path(m['output']).read_bytes() == raw
            else:
                assert not Path(m['output']).exists()
            same = d['encoding_device'] == d['decoding_device']
            if same:
                assert m['success'], (provider, case['fixture'], m.get('error'))
                counts['same_device_passes'] += 1
            else:
                bucket = 'cross_device_passes' if m['success'] else 'cross_device_failures'
                counts[bucket][provider] = counts[bucket].get(provider, 0) + 1
            if same or provider == 'vitni':
                assert m['success']
                assert d['encoder_decoder_trace']['cdf_sha256']['different_rows'] == 0
                assert d['encoder_decoder_trace']['logits_sha256']['different_rows'] == 0
        if provider == 'vitni':
            assert case['identical_archive'] and case['identical_payload']
            a, b = case['encodes'].values()
            assert Path(a['output']).read_bytes() == Path(b['output']).read_bytes()
            assert case['cpu_gpu_encoding_trace']['cdf_sha256']['different_rows'] == 0
            assert case['cpu_gpu_encoding_trace']['logits_sha256']['different_rows'] == 0
            counts['vitni_coded_positions'] += a['trace_rows']
    counts['vitni_compared_vocabulary_words'] = counts['vitni_coded_positions'] * 49152
    source = OUT/'cache/upstream/llmzip.py'
    raw = source.read_bytes()
    git_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert git_blob == '70813265196c70bf1dee5c3793aa586181de9810'
    assert 'selftest: ALL PASS' in (OUT/'upstream-selftest.log').read_text()
    assert 'research-prefill' not in (OUT/'build-features.txt').read_text()
    (OUT/'VERIFICATION.json').write_text(json.dumps(dict(passed=True, **counts), indent=2)+'\n')
    print(json.dumps(dict(passed=True, **counts), indent=2))


if __name__ == '__main__':
    main()
