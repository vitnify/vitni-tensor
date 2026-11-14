# llmzip CPU/Metal replay trial

Experimental adapter for [Andreas Lindeman's llmzip](https://github.com/andreasviner/llm-compression), using the existing Vitni v0.3.3 engine. Engine files are unchanged. This research branch adds only this experiment directory; it is not a default integration in either project.

## Try the short case

Requires an Apple Silicon Mac with Metal/MPS, Apple Command Line Tools, **Rust nightly** selected for `cargo`, and **Python 3.12**. The measured machine was an M3 Max running macOS 27.2.1. Other machines have not been validated.

```sh
git clone --depth 1 --single-branch --branch research/llmzip-portability-2026-11-14 https://github.com/vitnify/vitni-tensor.git vitni-llmzip-trial
cd vitni-llmzip-trial
python3.12 -m venv .venv-llmzip
.venv-llmzip/bin/python -m pip install -r experiments/llmzip/requirements.txt
.venv-llmzip/bin/python experiments/llmzip/prepare.py
.venv-llmzip/bin/python experiments/llmzip/try_vitni.py
```

Setup downloads about **414 MB of model weights**, plus tokenizer data and Python/Rust dependencies. It downloads the pinned upstream source and verifies its Git blob, model SHA256 hashes and fixture hashes. Upstream code and example text are fetched from their original repository, not vendored here. The build enables default features only; batched-prefill research code is absent from this branch.

The short trial uses the first 512 characters of upstream `examples/code_python.txt`. It independently encodes on CPU and Metal and runs four fresh decoders, including both cross-device directions. It compares the recovered file bytes, every full-logit/CDF hash and the two independent archives. Decoders receive archive and model paths, never expected plaintext or saved logits. Raw LZMA fallback is rejected.

Expected result on the measured pairing:

```text
PASS: all four fresh decoders recovered 512 exact bytes; CPU/Metal archives are identical (205 bytes).
```

Logs, archives, recovered files, per-position hashes and timings are saved under `experiments/llmzip/runs-quick/`. Each invocation uses a new directory. A failure exits unsuccessfully; it does not fall back to another backend.

To use a small UTF-8 file of your own:

```sh
.venv-llmzip/bin/python experiments/llmzip/try_vitni.py --input /path/to/example.txt
```

This is the original text codec. Non-round-tripping tokenization and non-UTF-8 input are not supported by this trial.

## Measured benefit and cost

The original 54-run experiment used pinned upstream commit `6e6cd1bccfa99584b8f112957c5a74f3f29b17a1`, SmolLM2-135M, complete upstream English prose (5,130 bytes), and fixed Python/Norwegian prefixes. Window=256, keep=64, arithmetic mode, temperature=1, no mixing. The upstream default window=1024 and default GPT-2 model were not tested.

| Backend | Same-device exact recovery | Cross-device exact recovery |
|---|---:|---:|
| Original HF weights, PyTorch FP32 CPU/MPS | 6/6 | 0/6 |
| Same Q8 GGUF weights as Vitni, PyTorch CPU/MPS | 6/6 | 0/6 |
| Vitni CPU/Metal | 6/6 | 6/6 |

Every ordinary cross-device failure reached the original integrity check. Every same-device control passed. Vitni produced byte-identical CPU/Metal archives for all three inputs. Full-logit and original integer-CDF hashes matched at all 1,689 coded positions (49,152 vocabulary words each).

For the full 5,130-byte prose file:

| Backend | CPU encode work | GPU encode work | Complete archive |
|---|---:|---:|---:|
| Original HF PyTorch | 19.05 s | 16.24 s | 750 B |
| Same-Q8 PyTorch | 33.66 s | 21.43 s | 804 B |
| Vitni | 115.88 s | 38.09 s | 828 B |

Vitni's observed GPU encode cost was **2.35× original HF PyTorch**; CPU was **6.08×**. All three prose payloads were 668 bytes long; the full-size difference comes from profile headers. Equal payload length does not mean equal payload contents or compatible prediction profiles.

These are **single-run observations on a desktop that remained in use**, not controlled performance ratios. Work time excludes imports/loading, download and compilation; cold-process GPU encode elapsed time was 42.45 s for Vitni versus 20.18 s for native HF. The original CLI normally decodes to verify a compression, adding a second computation: measured encode plus same-device decode work was about 76.21 s for Vitni GPU versus 32.53 s for native HF GPU. Full timings and variable-load caveats are in [RESULTS.txt](RESULTS.txt).

Both endpoints need the matching **144,810,560-byte GGUF model**, tokenizer and runtime. Those bytes are additional to the archive sizes. This assumes a shared model; it is not a one-file total-storage saving.

## Validation limits

- One physical M3 Max CPU/Metal pairing, one pinned model and software stack. No CUDA, AMD, x86 or cross-machine validation; no other models, long-corpus or library-version guarantee.
- The original CDF still uses CPU NumPy float64 math. Its behavior across different NumPy/libm builds is unvalidated, even with matching Vitni logits.
- This is a **new numerical profile**. Existing native-HF archives require their original profile and must be re-encoded for Vitni. The adapter rejects an incompatible profile before inference.
- The host filesystem is not isolated from decoder processes. The code avoids reading source files during decoding; this is a functional test, not a malicious-decoder sandbox or cryptographic execution proof.
- These are our tests on an independent workload. Owner interest, independent replication, adoption and commercial demand are unconfirmed.

## Full matrix and original evidence

The full matrix independently encodes on both devices, then decodes every archive on both devices for all three providers and inputs. It takes much longer than the short trial. Completed cases are resumed from `runs/`; use a fresh directory/clone for a fresh matrix.

```sh
.venv-llmzip/bin/python experiments/llmzip/run_matrix.py
.venv-llmzip/bin/python experiments/llmzip/analyze.py
.venv-llmzip/bin/python experiments/llmzip/verify.py
```

`summary.json` then describes the newly executed matrix. The historical results are stored separately:

- [Full original report](RESULTS.txt), [method](METHOD.txt) and [frozen plan](PLAN.json).
- [Original measurements](evidence/original-measurements.json), with local filesystem paths replaced by relative labels.
- [All original per-position hashes, gzip JSON](evidence/original-traces.json.gz).
- [Original profile/digest rejection controls](evidence/negative-controls.json) and [verification](evidence/VERIFICATION.json).

The original measurement checkout was `b43ba153c391364c224ef364adc8a6f84c9d1a17`, whose relevant arithmetic matches v0.3.3 when its gated `research-prefill` feature is disabled. This published reproducer starts directly from released v0.3.3 (`854a392ec81fc80c763d44efda2b45752853b559`); it adds no engine changes. The packaged short trial was separately tested against this release export before publication.
