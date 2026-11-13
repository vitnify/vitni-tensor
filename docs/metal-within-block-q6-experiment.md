# Within-block Q6 parallelism under lower load

2026-11-12. Research branch `research/metal-canonical-row-parallel-2026-11-11`.
Baseline commit `39f96e5dd322115451010dfa4a54607aa9f091b3`. Main remains
`b4d7e6456bef8b7f34c554ca993a41bebe17e317`.

The four-lane-per-block prototype preserved exact CPU bits and reduced median
TinyLlama forward time by 2.6193%, winning all three pairs. It missed the
prespecified 3% adoption gate. The prototype, tests, full integration patch and
results are archived here; the active engine has been restored byte for byte
to the optimized baseline. No defaults, main, push or release change.

## Mechanism and comparison

The retained Q6 kernel assigns eight physical threads to a row. Each thread
decodes and dots one complete 256-weight super-block before combining block
values with the canonical floating-point tree.

The prototype assigns four lanes to each super-block and eight super-blocks
to a 32-lane SIMD row. Each integer lane handles one `(half, pass)` pair:
sixteen positions in each of four quantization groups, or 64 weights. It forms
four scaled integer dots and combines four lane partials using two XOR
shuffles. Only afterward does the block's leader compute the original
`d_w * dx * float(acc)` with contraction off. Strided SIMD shuffles reproduce
the original adjacent-pair float tree, including odd tails without zero padding.

Every grouping of valid scaled integer products fits in signed int32:

```text
256 weights × 32 max |Q6 integer| × 127 max |activation integer|
            × 128 max |signed scale| = 133,169,152 < 2^31
```

One lane's partial is at most 33,292,288 in magnitude. Integer reassociation
therefore does not introduce overflow or rounding. The unchanged float tree
still matters after the integer partials are combined. This bound does not
claim equivalence for malformed inputs outside the engine's quantization domain.

Quantize-once, weights, activation bytes, output/residual semantics and model
storage are unchanged. The candidate uses 32 rather than eight threads per
hidden output row, with 64-thread groups. The original classifier remains on
its existing kernel. Domain: K divisible by 256, one through 64 super-blocks,
32-lane SIMD. Only the local Apple M3 Max was tested; no CUDA/other-GPU claim.

The baseline is the **current cooperative Q6**, not the older one-thread row
kernel. Whole-model controls retain metadata Q4, RMS `no_contract`, cooperative
RMS/Q6/attention, context 64, regime 2 and the original classifier. The immutable
baseline predictor SHA256 is
`832c37d41786eeeee8d0fb1191ae97490b86ea9f400cc24a0b630248efd836a5`.
The integrated candidate predictor SHA256 is
`bcf0d6771ff4d7597af4bb1fed4e7e30922e5548063c57e2f9dc6c2b096a2e24`.
Readiness asserts the actual candidate selector and 20 eligible hidden matrices
for TinyLlama, two for Stories. No silent fallback was accepted.

## Load qualification and stop rules

Before each kernel cohort, full-model pair and ordinary-control worker, require
two consecutive one-minute load averages below 7 at least ten seconds apart.
Wait at most 180 seconds, and abort without benchmarking if the gate fails.
Record before/after worker samples; any worker straddling a reading of 7 or
higher cannot support a lower-load timing claim. No background apps were closed.

The pilot entered at 5.22 / 5.69 and ended at 5.47. The nine-batch confirmation
entered at 5.08 / 4.53 and ended at 4.53. Every full-model and ordinary-control
worker also passed its before/after gate. These are CPU load averages, not proof
of zero GPU contention or fixed GPU frequency.

Stop for any numerical mismatch. Stop a kernel screen with >5% summed GPU
regression and at least 75% fixture losses. A passing/mixed screen receives a
bounded whole-model test; after its first pair, stop for >10% forward or >15%
fresh-worker regression. Adoption still requires >=3% median forward reduction,
three paired wins, <=2% median fresh-worker regression and the additional
certificate/regime/codec checks. The gate was fixed before timing the candidate.

## Real-computation kernel results

Eight pinned CPU captures cover TinyLlama hidden value/down projections at
steps 0, 7 and 63, plus two Stories down projections. Previously recorded SHA256
manifests were verified. Both overwrite and residual results, quantization
scales/int8 values and padded guards are checked against CPU, not just against
the previous GPU implementation.

Each alternating-order batch measures sixteen quantize-plus-dot dispatch pairs
using GPU command timestamps, with resident buffers. Compilation/allocation are
outside the kernel timer; the unchanged quantization pass is inside it. Host
time and every batch observation are saved. No warmup result is substituted for
a measured batch.

The three-batch pilot reduced the sum of fixture GPU medians by 7.6694%, with
seven of eight wins. The nine-batch confirmation reduced it by 5.3653%, again
with seven wins:

| Fixture | K × M | Baseline GPU µs | Candidate GPU µs | Change |
| --- | --- | ---: | ---: | ---: |
| Tiny step 0 value | 2048 × 256 | 67.000 | 57.461 | -14.24% |
| Tiny step 0 down | 5632 × 2048 | 97.552 | 107.406 | +10.10% |
| Tiny step 7 value | 2048 × 256 | 21.682 | 18.708 | -13.72% |
| Tiny step 7 down | 5632 × 2048 | 70.180 | 63.719 | -9.21% |
| Tiny step 63 value | 2048 × 256 | 19.922 | 17.737 | -10.97% |
| Tiny step 63 down | 5632 × 2048 | 69.568 | 64.539 | -7.23% |
| Stories step 0 down | 768 × 288 | 19.023 | 16.828 | -11.54% |
| Stories step 7 down | 768 × 288 | 19.099 | 17.023 | -10.87% |

These are per-fixture medians, not model-frequency-weighted savings. The first
down fixture regressed and remains in the aggregate. Frequency/cache effects
and small-dispatch overhead make a full-model check necessary. There was no
shape-specific rescue, lane-count search or additional cohort after the final
below-gate result.

## Complete model results

Three fresh serial TinyLlama pairs alternate order baseline/candidate,
candidate/baseline, baseline/candidate. Each runs the same pinned 108-token
history, including its reset. Every raw logit byte is checked against independent
CPU files. Three ordinary-Metal control runs construct fresh model instances in the same
Python orchestrator, using the same model, tokens, context and one-token batches.
They are performance controls, not exactness controls or fresh-process startup
comparisons. Baseline/candidate native workers are separate fresh processes.

| Pair | Baseline forward seconds | Candidate forward seconds |
| --- | ---: | ---: |
| 0 | 1.453287296 | 1.409819752 |
| 1 | 1.454077625 | 1.415220921 |
| 2 | 1.446516380 | 1.418633292 |
| Median | **1.453287296** | **1.415220921** |

The candidate wins all three pairs; the median forward reduction is **2.6193%**.
Median fresh-worker time, including setup, protocol and checking, falls from
2.966134292 to 2.902286083 seconds (2.1526%). The first candidate worker is
slower overall: 3.316146125 versus 3.123104291 seconds, including setup of
0.986083917 versus 0.683899625. Later setup observations are also retained;
the median does not erase this first-worker cost.

Fresh ordinary forward observations are 0.506928292, 0.494651540 and
0.522130620 seconds, median 0.506928292. In this cohort the retained baseline
is 2.86685x ordinary Metal and the candidate 2.79176x. Do not combine these
denominators with earlier 3.29x or 2.84x reports.

A 64-token second-network pair validates numerical behavior on another topology.
Its timings are saved but are not a second-network speed claim: there is only
one pair, with baseline first. Both networks' candidate selectors are asserted.

## Exactness and decision

- All 12,416,000 candidate logit words matched CPU exactly: three 108-step
  TinyLlama runs and one 64-step Stories run. Baseline controls independently
  matched the same number of words. These are complete forwards, including
  actual residual use, attention state and resets.
- Each real-fixture mode also matched 14,976 overwrite/residual scalar outputs
  and guards. Every timed batch's final overwrite output matched CPU.
- Three conformance tests passed with zero ignored/skipped tests. GPU boundary
  checks cover block counts 1, 2, 3, 7, 8, 9, 22, 31, 32, 33, 56, 63 and 64,
  zero/signed-zero and varied inputs, and row tails. Extreme tests exercise
  Q6 integers -32/+31, activations +/-127, signed scales -128/+127 and mixed
  cancellation. Together they check 3,216 scalar outputs across both modes,
  plus exact quantization and output guards. A topology test checks the float
  tree for every block count 1..64.

The measured gain is positive but misses the preset 3% whole-history gate. The
prototype is archived rather than promoted into the optimized baseline. This
does not establish no possible benefit from other lane layouts, nor validate
larger models, long contexts or other hardware. No new certificate, regime
switch or bidirectional codec campaign followed because adoption failed before
those required checks. Finite conformance tests are not an exhaustive proof.

## Reproduction and artifacts

The archived standalone GPU tests compile baseline kernels plus the test-only
fixture; they do not change active dispatch:

```sh
cargo test --manifest-path metal/Cargo.toml --test within_block_q6k -- --include-ignored --nocapture
```

`docs/research/within-block-q6-trial.patch` restores the exact opt-in whole-model
prototype (`VITNI_METAL_Q6K_WITHIN_BLOCK=1`, with cooperative Q6 enabled). It
passes `git apply --check` against the restored research baseline. Applying it
is a research experiment, not adoption of a validated default.

The local parent-workspace artifact directory is
`outputs/vitni-metal-q6-within-block-2026-11-12`. It contains the plan, archived
binaries, exact candidate/base source snapshots, full patch, Cargo lockfile,
kernel observations, all per-token CPU checks and readiness metadata, load
samples, ordinary controls, scripts, logs and SHA256 provenance. The no-feature
kernel probe binary used before the full-model build is preserved from Cargo's
matching dependency artifact; the later feature build is not substituted for
the measured binary. The committed numeric summary is
`docs/research/within-block-q6-2026-11-12.json`.
