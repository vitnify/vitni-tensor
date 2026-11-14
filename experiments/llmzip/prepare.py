"""Fetch pinned upstream/model inputs, verify hashes and build the adapter.

No engine or upstream source is patched. First setup downloads about 414 MB of
weights plus tokenizer data, in addition to Python/Rust dependencies.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

OUT = Path(__file__).resolve().parent
COMMIT = '6e6cd1bccfa99584b8f112957c5a74f3f29b17a1'
HF_REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
GGUF_REVISION = '1076b85d135017971845b19b8d07df1479155988'
GGUF_SHA = '813966e2b5e5aba261f6f198849a13b815db3c77bf4e389e72aa1e0f35a0d455'
HF_SHA = '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
FIXTURES = {
    'prose_full': ('prose_en.txt', None, '082bf9d8fdda1eef51d9a4a5d0805baba9167b7ec10cf8e76c9fe0dd2b8b14fb'),
    'code_prefix': ('code_python.txt', 512, 'acb0d7663ba44624dc730b52db6b42c6749454110a469b146112cd47c56359d1'),
    'norwegian_prefix': ('prose_no.txt', 512, '9bc9fbe5f8dc899a75fb957e9801a749e11114e3ee28287c0dd28de63fa29599'),
}


def file_info(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=digest, bytes=path.stat().st_size)


def main():
    if sys.version_info < (3, 12):
        raise SystemExit('Python 3.12 or newer required; tested with 3.12.11')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-cache', type=Path, default=OUT/'cache/hf')
    parser.add_argument('--upstream-archive', type=Path,
                        help='optional predownloaded tarball of the exact upstream commit')
    parser.add_argument('--offline-build', action='store_true',
                        help='require Rust dependencies already in the Cargo cache')
    args = parser.parse_args()
    cache = OUT/'cache'
    cache.mkdir(exist_ok=True)
    source = cache/'upstream'
    source.mkdir(exist_ok=True)
    if args.upstream_archive:
        archive = args.upstream_archive.read_bytes()
    else:
        url = f'https://api.github.com/repos/andreasviner/llm-compression/tarball/{COMMIT}'
        request = urllib.request.Request(url, headers={'User-Agent': 'vitni-llmzip-reproducer'})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read()
    # Only extract the codec and fixtures required for this experiment.
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for member in tar.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            allowed = relative == Path('llmzip.py') or relative.parent == Path('examples')
            if not allowed or not member.isfile() or '..' in relative.parts:
                continue
            target = source/relative
            target.parent.mkdir(exist_ok=True, parents=True)
            with tar.extractfile(member) as stream:
                target.write_bytes(stream.read())
    raw = (source/'llmzip.py').read_bytes()
    git_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if git_blob != '70813265196c70bf1dee5c3793aa586181de9810':
        raise RuntimeError('upstream source does not match the pinned Git blob')
    fixture_dir = OUT/'fixtures'
    fixture_dir.mkdir(exist_ok=True)
    for name, (filename, characters, digest) in FIXTURES.items():
        original = (source/'examples'/filename).read_bytes()
        data = original if characters is None else original.decode('utf-8')[:characters].encode('utf-8')
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'fixture hash mismatch: {name}')
        (fixture_dir/f'{name}.txt').write_bytes(data)
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot = snapshot_download('HuggingFaceTB/SmolLM2-135M', revision=HF_REVISION,
                                 cache_dir=str(args.hf_cache.resolve()),
                                 allow_patterns=['*.json', '*.safetensors', '*.txt'])
    gguf = hf_hub_download('ggml-org/SmolLM2-135M-GGUF',
                          'SmolLM2-135M-Q8_0.gguf', revision=GGUF_REVISION,
                          cache_dir=str(args.hf_cache.resolve()))
    files = {'SmolLM2-135M-Q8_0.gguf': file_info(gguf),
             'model.safetensors': file_info(Path(snapshot)/'model.safetensors')}
    if files['SmolLM2-135M-Q8_0.gguf']['sha256'] != GGUF_SHA:
        raise RuntimeError('GGUF hash mismatch')
    if files['model.safetensors']['sha256'] != HF_SHA:
        raise RuntimeError('HF weights hash mismatch')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json']:
        files[name] = file_info(Path(snapshot)/name)
    (cache/'models.json').write_text(json.dumps(dict(
        gguf=str(Path(gguf).resolve()), hf_snapshot=str(Path(snapshot).resolve()),
        gguf_revision=GGUF_REVISION, hf_revision=HF_REVISION, files=files), indent=2)+'\n')
    manifest = str(OUT/'native/Cargo.toml')
    build = ['cargo', 'build', '--locked', '--release', '--manifest-path', manifest,
             '--target-dir', str(OUT/'native/target')]
    if args.offline_build:
        build.append('--offline')
    subprocess.run(build, check=True)
    with (OUT/'build-features.txt').open('w') as stream:
        subprocess.run(['cargo', 'tree', '--locked', '--offline', '--manifest-path', manifest,
                        '-e', 'features'], stdout=stream, check=True)
    with (OUT/'upstream-selftest.log').open('w') as stream:
        subprocess.run([sys.executable, str(source/'llmzip.py'), 'selftest'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('Pinned inputs verified, upstream selftests passed, release adapter built.')
    print('Next: python experiments/llmzip/try_vitni.py')


if __name__ == '__main__':
    main()
