# Metal SiLU overflow correction and validation

Release note: the tested implementation recorded here ships in v0.3.3.
Research-branch status statements below describe the original measurement snapshot.

Date: 2026-11-13. Branch: `research/torch-full-model-2026-11-13`.
Baseline: `e773662b6dde7c73b0f1401819d4892a01daf0cd` (`e773662`, the tested resident wrapper).
This change remains on the research branch, unmerged and unpublished.

The fixed Metal path follows the existing CPU SiLU/SwiGLU expression and
rounding in the tested finite-input cases. The CPU implementation, numerical
regime and published CPU anchors are unchanged. The original optimized
baseline is preserved as immutable shader, worker and native-library files;
their hashes are in the accompanying JSON.

## Root cause and fix

The CPU computes `x / (1 + libm::expf(-x))`. For sufficiently negative finite
`x`, the exponential overflows to infinity and the division produces signed
zero. The old Metal reciprocal seed/refinement did not handle an infinite
divisor. A first narrow guard fixed that case, but dense boundary tests exposed
two related problems: the reciprocal also becomes invalid/subnormal for very
large finite divisors, and Metal flushes some subnormal activation/product
values. The preliminary failed checks are preserved.

The correction has three parts:

1. Shared `div_sw` returns the exact sign bit for a finite numerator divided
   by either infinity, bypassing the reciprocal approximation.
2. SiLU scales both division operands by exactly `2^-64` when its finite
   denominator is at least `2^120`. In this SiLU range both scaled operands
   and the quotient remain normal. Tiny inputs use a ties-to-even bitwise
   half, because their rounded exponential is exactly one and denominator two.
3. SwiGLU keeps ordinary normal products on hardware. Subnormal inputs or
   potential underflow use an exact 24x24-bit integer product and explicit
   nearest-even float32 packing, preserving signed zero and subnormal results.

Both active `silu_kernel` and `silu_mul` use these helpers. The inactive fusion
fixture now uses the same activation helpers; it remains disabled. The CPU's
overflow-to-zero behavior is retained rather than changing its numerical
formula or moving its anchors. Non-SiLU normal division arithmetic is unchanged.

## Regression coverage

Exact GPU comparisons against the actual CPU SiLU operation or native CPU
float32 multiplication passed:

- 49,036 SiLU outputs, including a dense 16,385-word neighborhood of the
  exponential overflow boundary, signed zero, tiny inputs, maximum magnitudes
  and deterministic finite-bit-pattern samples.
- 539,396 SwiGLU outputs across finite multiplier signs and magnitudes.
- 99,446 independent product-rounding cases, including normal/subnormal
  transitions, ties, signed zero and finite-input product overflow.
- 28 finite-over-infinity divisions, with both denominator and zero signs.

Output guard words were checked. The frozen old shader had six mismatches in
six dedicated large-negative cases; the fixed shader had none. The existing
real-matrix overflow fixture also now matches CPU. All **16 Metal tests**
passed, including the retained Q4/RMS/Q6/attention and regime-switch tests.

## Complete-model counterexample and replay

A separate copy of TinyLlama modifies only the two half-scale coefficients
in each block of `blk.0.ffn_gate.weight` by an exact factor of 128. All
90,112 modified coefficients remained
finite and exactly representable. The original model file was not changed.
This is a fault-injection checkpoint, not a quality or throughput claim.

For token 1, the actual CPU first-layer gate computation has
1 exponential overflow and
2 large finite exponential
cases. Gates range from -143.48819 to
178.62155; CPU logits remain finite.
The old optimized engine returned finite but incorrect logits:
**128,000/128,000 words
differed from CPU**. The fixed installed wheel matched all 128,000
words exactly, and its complete generated receipt matched CPU and the frozen
CPU receipt executable. A finite-output check alone would not detect the old
wrong answer.

On the original unmodified models, the final installed wheel also matched
all **24,320,000 CPU/Metal logit words**: 3,840,000 from independent pinned
CPU captures plus 20,480,000 from maximum-context replay (512 TinyLlama and
128 Stories positions). Complete receipts for lengths 0, 4 and 16 matched
across CPU/Metal; length 4 also matched the frozen standalone CPU/Metal CLIs.
These checks compare exact words and complete schemas, not tolerances or
selected output tokens. The installed package passed all **57 tests**.

## Runtime against the optimized baseline

Nine rotating/reversed-order rounds per case, after two warmups, with three
resident instances: frozen optimized Rust worker, fixed optimized Rust worker,
and fixed PyTorch wrapper. All use the retained metadata/RMS/Q6/attention/
concurrent configuration. The benchmark runs serially. Each timed forward
compares all logits across all three variants; generation compares full
receipts. No corrected stress-checkpoint timing is mixed into these results.

Native timings exclude process IPC, file writes and preallocated full-logit
output. PyTorch wall time includes allocation, validation, FFI and conversion.
Model loading and shader compilation are excluded; GPU work completes before
the timer ends. Hardware: M3 Max, 14 CPU cores, 36 GiB; PyTorch 2.14.1.

| Model | Workload | Frozen optimized | Fixed optimized | Fixed change | Fixed PyTorch |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 64-token forward | 0.724738 s | 0.714166 s | -1.459% | 0.715266 s |
| TinyLlama 1.1B | 4-token prompt + 32 generated | 0.403475 s | 0.403791 s | +0.078% | 0.402563 s |
| Stories 15M | 64-token forward | 0.170261 s | 0.169897 s | -0.214% | 0.171001 s |
| Stories 15M | 4-token prompt + 32 generated | 0.096765 s | 0.096857 s | +0.096% | 0.097061 s |

The generation medians changed by about +0.08% (TinyLlama) and +0.10%
(Stories). Forward medians were modestly lower. This cohort shows no material
runtime regression; it does not establish a causal speedup from the fix.
Raw repetitions, load observations, actual optimization flags and binary
hashes are preserved. The older approximately 2.47x comparison to ordinary
llama.cpp is a different benchmark and was not remeasured here.

## Reproduction and scope

Build/install and full-model use follow [the guide](../../python/FULL_MODEL.md).
Run the native regression suite on an actual Metal device:

```sh
cargo test --release --manifest-path metal/Cargo.toml \
  --test silu_overflow --test q4_prepare_swiglu -- --ignored --nocapture
```

For the old/fixed scalar contrast, set `VITNI_SILU_BASELINE_SOURCE` to the
frozen shader, or extract it from `e773662:kernels/forward_ops.metal`.
The normal fixed-shader tests do not require that optional comparison file.
Strict installed-package checks require `VITNI_REQUIRE_MPS=1`,
`VITNI_REQUIRE_FULL_MODEL=1` and `VITNI_FULL_MODEL_GGUF` as documented in
the guide. The workspace artifact `outputs/vitni-silu-overflow-2026-11-13`
preserves checkpoint construction, gate counts, baseline/fixed logits,
all replay/benchmark drivers, frozen binaries and build/test logs.

This fixes the identified finite-input Metal activation cases. It does not
claim general IEEE-754 division, cross-vendor NaN-payload semantics, physical
CUDA validation, or equivalence for every model/GPU/activation pattern.
The full-model API remains eager and unbatched, with CPU tensor I/O and a
512-position maximum (or the smaller model limit).

[Machine-readable evidence](silu-overflow-2026-11-13.json).
