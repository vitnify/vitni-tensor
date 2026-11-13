# Separate Q4 compiler-limit and RMS rounding-control experiments

November 12, 2026. Baseline: retained metadata Q4 with contraction disabled,
cooperative RMS, hidden Q6 and attention, and the original classifier. Both
candidates were tested independently. Main and default behavior are unchanged.

## Q4 compiler limit: stopped at the pilot

Keep the existing Q4 metadata shader, rounding, weight format and 64-thread
dispatch. Compare the current function-based pipeline constructor against a
descriptor with its automatic maximum, and against that same descriptor with
`maxTotalThreadsPerThreadgroup=64`. Observed pipeline maxima are 1024 for both
automatic controls and 64 for the candidate; SIMD execution width remains 32.

The CPU differential boundary check passes 2,088 scalar comparisons including
controls, plus padded-row guards. Real captured TinyLlama/Stories15M matrix
outputs also match CPU in overwrite and residual forms.

Three alternating batches of sixteen resident dispatches per real input give
the following summed TinyLlama matrix GPU-time ratios:

| Configuration | Ratio to existing constructor |
| --- | ---: |
| Existing function-based constructor | 1.00000 |
| Descriptor, automatic maximum | 1.06127 |
| Descriptor, maximum 64 | 1.06449 |

The candidate loses on three of four TinyLlama inputs and crosses the declared
early-stop screen: over 5% summed cost with losses on at least 75% of inputs.
Stop its performance campaign. The extra descriptor control matters: the cap
is only 0.30% slower than descriptor-auto in these short observations. The
results do not attribute the whole 6.45% penalty to the thread limit itself,
prove a register-spill cause, or establish a stable production slowdown.
No full-model speed/replay claim is made for this stopped candidate. The
active engine retains its previous pipeline constructor; the rejected probe,
all timings and source remain archived.

## RMS local contraction control: retain behind an opt-in

At the beginning of the cooperative RMS kernel's compound statement, use
`#pragma METAL fp contract(off)`. Replace only the square-product and scaled-value
`volatile` intermediates with ordinary floats. Preserve the eight canonical
chains, adjacent-pair tree, mean, square root, software division, explicit FMAs
inside the helper, `(x*scale)*weight` operation order, barriers and output loop.
No helper, Q4, Q6 or attention arithmetic changes. Fast math stays disabled.

A contraction-sensitive control returns separately rounded zero while explicit
`fma` still returns `0xa8800000`, confirming the intended compiler behavior.
Boundary tests pass 1,974,528 scalar output comparisons including both variants,
plus guards, with odd widths/rows, widths through 8192, three epsilons, signed
zeros, varied magnitudes and both in-place/out-of-place writes.

### Actual normalization inputs

Capture real CPU residuals and RMS weights at steps 0 and 7, first/middle/last
layers, for attention and FFN normalization in each of TinyLlama and derived
Stories15M: twelve inputs per model. Use actual `Tensor::rms_norm` outputs as
references. All sixteen CPU forwards used during capture match pinned original
raw logits. No synthetic input substitutes for these model measurements.

The initial three-batch RMS pilot reduces summed TinyLlama GPU time by 35.70%.
The nine-batch follow-up reduces it by 28.24%; the second network's follow-up
reduction is 21.37%. These are resident kernel timings with allocation and
compilation excluded, not complete-model gains.

### Complete matched-history result

One M3 Max, TinyLlama 1.1B Chat Q4_K_M, unchanged frozen 108-token codec history,
original positions/two resets, context 64, regime 2. Three fresh serial workers
per Vitni mode, reversed order on alternating repetitions. Every raw logit byte
compares directly with independent CPU. Three fresh ordinary Metal controls use
the same inputs. The lower-load start and before/after gates pass; no background
application is stopped or changed and no observation is discarded.

| Median seconds | Optimized metadata baseline | Plus RMS contraction control |
| --- | ---: | ---: |
| Forwarding | 1.508254 | 1.448267 |
| Fresh worker, including setup/protocol/verification | 3.043292 | 2.939159 |
| Setup | 0.588381 | 0.570208 |

Forward time falls 3.9772%, with three paired wins. Median fresh-worker time
falls 3.4217%. The first candidate worker is slower: 3.240371 s versus
3.082614 s, including 0.883599 s setup. It remains in the samples. These are
fresh processes with normal system caches, not forced cache-cold measurements.

Ordinary Metal's median is 0.505800 s: same-cohort ratios are 2.9819× baseline
and 2.8633× candidate. The previous 3.29× used another control measurement;
the full cross-session ratio change is not attributed to this RMS change.

The RMS change adds no weight or activation buffers. Existing metadata Q4's
storage expansion remains: it is not removed by this experiment. Shader code
and pipeline-object memory, peak RSS and production memory behavior were not
measured.

### Replay and recovery

- 12,416,000 candidate full-history logit values match independent CPU bytes:
  three 108-step TinyLlama histories plus one 64-step derived Stories15M history.
  Baseline controls pass another 12,416,000 values. No tolerance/top-token-only
  substitution; full saved hashes revalidate against pinned CPU files.
- The permanent RMS GPU boundary test passes; actual model 2→3→2 switches and
  resets pass 768,000 direct logit-word comparisons across the two networks.
- Full CPU/Metal certificate agreement and the regime-3 regression pass.
- Fresh current/candidate ASCII encodes match the CPU cover/key, coding tables
  and full-logit-row fingerprints. CPU→candidate Metal and candidate Metal→CPU
  decoding both recover the exact original file. Decoders receive only cover/key.
  Codec timings have one observation per mode and are correctness checks.

The first RMS timing harness failed in the baseline's guard initialization
before recording timings: an untyped literal produced an f64 initialization
buffer for an f32 GPU buffer. It was corrected to f32; no kernel changed and
the failed source/binary/log were retained. The earlier independent CPU boundary
tests already passed. A reporting-macro compile error was also fixed before
execution. No measured result was removed or replaced. The final retained shader
differs from the measured shader only by a stripped trailing blank line; both
source snapshots and the measured binary are retained, and final recovery checks
use the rebuilt retained binary.

## Opt-in and scope

Before constructing `MetalForward`, enable:

```sh
VITNI_METAL_Q4K_VARIANT=metadata
VITNI_METAL_RMS_VARIANT=no_contract
VITNI_METAL_Q6K_COOPERATIVE=1
VITNI_METAL_ATTENTION_COOPERATIVE=1
```

The RMS variant also enables cooperative RMS for supported widths 1..=8192.
`VITNI_METAL_RMS_VARIANT=current` or an unset variable preserves the existing
choice; unsupported widths keep the existing scalar path. The getter reports
the actual selected RMS variant. Q4 pipeline limits remain automatic.

Keep the change research-only and opt-in. One physical Mac/compiler and one
performance workload, with a second small topology for exactness, do not validate
other GPUs, larger models, long contexts, arbitrary IEEE-754 inputs, native speed
parity or production concurrency. No new arithmetic regime or novelty claim.

Artifact: `outputs/vitni-metal-threadcap-rms-rounding-2026-11-12` in the research
workspace. It preserves plans, immutable baseline/candidate provenance, both
isolated probes, all pilot/full observations, captured inputs, CPU-checked rows,
codec recoveries, failed attempts, source patches and manifests. Compile offline
with the existing Cargo cache; GPU execution requires host access. From the repo:

```sh
cargo test --release --offline --manifest-path metal/Cargo.toml \
  --test rms_rounding -- --ignored --nocapture --test-threads=1
```

Full-model checks additionally require `VITNI_GGUF` and the opt-ins above.
Exact worker commands and environments are retained in the artifact. No main
change, default promotion, push, release or paid compute occurred.
