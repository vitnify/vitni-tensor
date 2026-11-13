# Resident full-model PyTorch wrapper: research validation

Date: 2026-11-13. Branch: `research/torch-full-model-2026-11-13`.
Base: `v0.3.2` (`9cc5ac50671bcdbee144d3fb1214fe87b63e6a55`).
The Python research package is `vitni-torch` 0.1.0a2. This work remains
local to the research branch and is not merged or published.

The eager `VitniModel` module now calls the existing full engine. It keeps a
resident native owner instead of rebuilding a network from the individual
PyTorch operators. On this Mac, its final Metal runtime tracks the resident
Rust control to within 1% in the tested median timings. All **24,320,000 raw
logit words** in the final cross-backend replay campaign matched exactly.

## Design

`VitniModel(gguf, backend='metal', optimized=True)` owns a process-local
native handle through a Rust C ABI. The handle registry stores per-model
`Arc<Mutex<Session>>` owners. Python also serializes calls to an instance.
Explicit close/context exit and a weak-reference finalizer release the owner;
closed, serialized and forked handles cannot be used as portable model state.
No libtorch C++ ABI is linked.

Metal uses the actual `MetalForward` with its existing resident weights,
Q4 metadata, scratch buffers and KV cache. Explicit `MetalOptions` select the
retained Q4/RMS/Q6/attention/concurrent configuration without changing global
optimization environment variables. The older `MetalForward::new` still
reads its environment through an adapter, covered by the existing regression
suite. No kernel source or numerical regime was changed.

CPU retains the GGUF bytes and KV cache, rebuilding borrowed weight views
per request without file I/O or matrix copies. This avoids unsafe
self-referential ownership. Each request clears/rebuilds sequence state and
reuses resident allocations. It is replay of a token history; live KV-state
migration is outside this API.

`forward` accepts a contiguous CPU int64 vector and returns owned CPU float32
logits for every position. `generate` calls the original generation and
certificate driver and returns its complete receipt schema. GPU execution is
native Metal; outputs are not tensors allocated on PyTorch's MPS stream.
The eager boundary is explicit. Training, `.to()`/dtype conversion,
`state_dict` and model pickling are rejected.

Metal constructors, requests and FFI cleanup use explicit Objective-C
autorelease scopes. This matters for a long-lived non-Cocoa host: command
buffers/encoders are autoreleased objects. After 100 warmup requests, a
700-call observation showed 0.80 MiB
of RSS growth over the next 600 calls, leveling off in the last samples.
This bounded observation is not a general leak proof.

## Final runtime comparison

Hardware: M3 Max, 14 CPU cores, 36 GiB, macOS ARM64; Python 3.12.11,
PyTorch 2.14.1. Benchmark/wheel compiler: Rust 1.95.0-nightly, with libm
pinned to 0.2.16. The separate CPU-only build also passed using stable Rust
1.97.1. The final timing binary hashes are in the accompanying JSON.

Nine alternating-order pairs per case, plus first calls and two warmups.
Both paths use the same resident engine/session and a fresh sequence per call.
Rust timers exclude IPC, file writes and the preallocated full-logit output
buffer. PyTorch wall time includes module dispatch, validation, allocation,
FFI, completed native work and result conversion. GPU completion is synchronous.

| Model | Complete workload | Resident Rust | PyTorch wall | PyTorch/Rust | Python boundary median |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 64-token full forward | 0.714344 s | 0.720188 s | 1.00818x | 97.2 us |
| TinyLlama 1.1B | 4-token prompt + 32 generated | 0.402448 s | 0.402640 s | 1.00048x | 83.8 us |
| Stories 15M | 64-token full forward | 0.169208 s | 0.169106 s | 0.99940x | 94.0 us |
| Stories 15M | 4-token prompt + 32 generated | 0.096413 s | 0.096209 s | 0.99788x | 70.3 us |

The boundary column measures Python wall minus the FFI's internal native
timer. Sub-percent cohort differences also include engine scheduling/clock
variation and do not establish a kernel speedup. Every timed forward compared
all output bytes; every timed generation compared complete receipts. Startup,
first calls, internal engine time, raw samples, actual optimization flags,
load observations and owner call counts are preserved in the JSON.

The control is a persistent Rust process calling the same engine directly,
not a fresh CLI process for every timed sample. Separate frozen v0.3.2 CLI
checks cover original standalone integration. The older approximately 2.47x
TinyLlama-versus-llama.cpp forward ratio is a separate experiment and was not
remeasured here. Earlier CPU timing cohorts are archived as supplementary
pre-lifetime-fix observations rather than final Metal performance evidence.

## Replay and independent references

- Original pinned CPU captures: 108 TinyLlama positions in two fresh sequences
  and 12 Stories positions in two sequences: **3,840,000
  words per backend**, with exact bytes matching on CPU and Metal.
- Maximum contexts: 512 TinyLlama and 128 Stories positions, with a deterministic
  integer-generated token history: **20,480,000
  words per backend**, zero mismatches. These checks used the final installed
  wheel after the autorelease fix.
- Complete CPU/Metal receipts for generation lengths 0, 4 and 16. For length 4,
  both backends also matched the unmodified, hash-pinned v0.3.2 standalone
  receipt binaries. Model identity, configuration, weights hash, regime,
  both digest versions and every generated token were checked.
- The CPU-only stable build independently matched another 128,000 original
  TinyLlama logit words and the complete receipt. An explicit Metal request
  correctly failed when Metal was not compiled; there was no CPU fallback.

Original TinyLlama output files were verified against their pre-existing
SHA256 fields. Stories records BLAKE3 rather than SHA256 for its output rows;
all 12 files were verified against those original BLAKE3 values, then assigned
SHA256 provenance for this artifact. Capture metadata itself was checked
against the earlier immutable input manifest. No reference bytes were changed.

An initial reference checker assumed both archives provided SHA256; that
assumption was corrected. A reporting-variable bug also omitted receipt
details after checking them; the comparison was rerun to save the details.
The final replay run uses the corrected recorder and includes all fields.
All preliminary logs remain in the workspace artifact.

## Test plan and outcomes

The final installed wheel passed **57 tests** on real CPU/MPS/Metal. This
includes the 40 existing individual-operator tests and 17 full-model cases:
raw logits/receipts, interleaved fresh requests, repeated use, output ownership,
invalid token/dtype/layout/context rejection, recovery after errors, serialized
concurrent calls, training/serialization/device contract rejection, close,
ambient flush-mode rejection and recovery, no subprocess inference, explicit
options ignoring ambient optimization variables, and inference after the
opened weight-file name is removed.

The nine retained Metal regression tests also passed, including metadata
regime switching through the existing environment constructor. The CPU-only
stable build passed seven selected full-model tests. The source archive was
built, extracted and rebuilt into a wheel; the installed native library,
Metal framework support, shader data, platform architecture and remapped
diagnostic paths were checked. No skipped model/GPU test is counted as proof.

To reproduce with a checkpoint, follow [the full-model guide](../../python/FULL_MODEL.md).
It provides package build, strict hardware/model test commands, maximum-context
CPU/Metal replay and the alternating standalone/PyTorch benchmark. Source:
[Python owner](../../python/vitni_torch/_model.py),
[native owner and ABI](../../bindings/torch/native/src/full_model.rs),
[resident Rust control](../../bindings/torch/native/src/bin/full-model-worker.rs).

## Limits

This is an eager, unbatched, inference-only GGUF bridge for the tested CPU and
Apple Metal engine. It does not convert arbitrary PyTorch/Hugging Face models,
provide gradients/CUDA, integrate PyTorch MPS allocator/streams, export native
handles, or migrate live KV state. Context is capped at the smaller of the
model limit and 512; Metal projection type/dimension checks match the released
CLI. The existing SiLU exponential-overflow edge remains unresolved, so these
finite-model tests do not establish universal IEEE-754 equality.

Larger models, other Apple GPU generations and physical Linux/CUDA execution
of this new full-model API remain unvalidated. The earlier individual
operators' PyTorch compilation/export tests remain applicable to those
operators; the full-model wrapper deliberately has an eager boundary.

[Machine-readable observations and hashes](torch-full-model-2026-11-13.json)
preserve the evidence. Workspace artifact:
`outputs/vitni-torch-full-model-2026-11-13`.
