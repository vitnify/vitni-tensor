# Separate concurrent-dispatch and RMS-to-Q4 fusion experiments

2026-11-12. Research branch `research/metal-canonical-row-parallel-2026-11-11`.
Both candidates start from `cfe8c718cfef21a7e5c15b6d90174ac241ea70b9`: metadata
Q4, RMS `no_contract`, cooperative hidden Q6 and attention, original classifier.
The archived within-block Q6 prototype is not enabled. Main remains
`b4d7e6456bef8b7f34c554ca993a41bebe17e317`.

**Retain concurrent dispatch behind an opt-in:** 12.5418% lower median complete
TinyLlama forward time, three paired wins, exact CPU output bytes and successful
certificate/regime/recovery checks. **Stop RMS-to-Q4 fusion:** 32.9552% higher
summed kernel GPU time and twelve of twelve losses, despite exact outputs.
The candidates were never combined. Default behavior remains serial; no push,
release or main change.

## Concurrent dispatch

The previous host creates Metal's default serial compute encoder even where
Q/K/V or gate/up are independent. The new flag
`VITNI_METAL_CONCURRENT_DISPATCH=1` selects an explicit concurrent encoder.
It changes scheduling, not shader arithmetic, grids, quantization, weight
formats, reduction order or persistent buffers. Necessary existing barriers
remain.

Shared activation scratch requires an additional dependency: an earlier Q6 dot
may still read `b_q8_dx`/`b_q8_qs` when a later quantization overwrites them.
Concurrent mode inserts a resource barrier before that reuse, in both overwrite
and residual paths. Regime 3 also protects `b_q8_bsums`. Existing post-quantization
barriers protect consumers from unfinished writes. The rest of the graph retains
its RMS, QKV, RoPE, attention, residual and FFN producer/consumer barriers.

`concurrent_dispatch_enabled()` reports configuration. The complete-history
predictor asserts it is true for the candidate and false for the immutable
baseline. Split-pass dispatch profiling is explicitly rejected with concurrent
mode: its different encoder structure would not measure this path. Command-level
profiling remains possible. No arithmetic kernel in `kernels/` changed.

Apple documents the [default serial encoder](https://developer.apple.com/documentation/metal/mtlcommandbuffer/makecomputecommandencoder%28%29)
and the synchronization responsibility for [concurrent dispatch](https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent).
This is use of a standard API, not an invention claim.

## Independent pilot and load gates

Apple M3 Max, 36 GiB. Before each timing cohort require two one-minute load
averages below 7 at least ten seconds apart, bounded to 180 seconds. Record
before/after samples, and do not qualify a worker crossing 7. Do not close
background apps. All reported pilot, confirmation, full-model and ordinary
control workers qualified. This does not establish zero GPU contention or fixed
GPU frequency.

Prespecified screen: three alternating batches of sixteen resident repetitions;
stop any bit mismatch, or >5% summed GPU regression with at least 75% fixture
losses. Otherwise extend to nine batches and a bounded complete-history cohort.
Stop that cohort after its first pair for >10% forward or >15% worker regression.
Adoption requires >=3% median full-history gain, three paired wins, <=2% median
fresh-worker regression, and additional certificate/regime/recovery checks.

Six independent real TinyLlama gate/up pairs cover steps 0 and 7 and layers
0, 11 and 21. Each projection has K=2048, M=5632. Both modes use the same packed
weights, input words, 64-thread groups and current metadata Q4 kernels. Baseline
dispatches the pair serially; candidate concurrently. An identical output barrier
in both modes prevents the next repetition overwriting unfinished output. CPU
words and padded guards are checked before timing and after every measured
batch. Compilation, packing and allocation are outside these kernel timers;
fresh-process costs are included separately below.

Concurrent dispatch reduces the sum of fixture GPU medians by 6.4793% in the
pilot and 8.2563% in the nine-batch confirmation. All six fixtures win both
screens. These kernel sums are not model-frequency-weighted forward savings.

## Complete-history result

Three fresh serial native-process pairs alternate baseline/candidate order.
Each executes the pinned 108-token TinyLlama history with its reset, context 64,
regime 2. The candidate asserts the optimized baseline flags and actual concurrent
selection. Every raw logit byte is compared to independent pinned CPU files.

| Pair | Baseline forward seconds | Concurrent forward seconds |
| --- | ---: | ---: |
| 0 | 1.449550874 | 1.273654914 |
| 1 | 1.439832976 | 1.261574538 |
| 2 | 1.442489008 | 1.254194799 |
| Median | **1.442489008** | **1.261574538** |

Median forward reduction: **12.5418%**, three of three paired wins. Median
fresh-worker time, including loading, protocol and checking, falls from
2.926154166 to 2.752340667 seconds, **5.9400%**. Median setup is 0.595968708
versus 0.598098375 seconds.

The first candidate worker is slower overall: 3.174459875 versus 2.989162375
seconds. Its setup is 0.990739375 versus 0.617980083 seconds. That first-worker
cost is retained in the observations; the median gain does not erase it.

Three fresh ordinary-Metal control processes run the same model, tokens, context
and one-token batches. Forward times are 0.511604417, 0.504920789 and 0.515897336
seconds, median 0.511604417. This cohort's ratio changes from **2.81954x** ordinary
for the baseline to **2.46592x** for concurrent dispatch. Ordinary outputs are
allowed to differ numerically. Do not pool this denominator with older ratios.

A complete 64-step derived Stories15M history also matches CPU under both modes.
It is a second-topology numerical check, not a second-model speed claim. The
full histories compare **12,416,000 raw words per mode** against CPU. Three fresh
TinyLlama candidate processes and the Stories candidate all assert concurrent
dispatch; no serial fallback is accepted.

## RMS-to-Q4 fusion: stop after the screen

The separate fixture computes the exact canonical RMS normalization in each
Q4 consumer workgroup, writes normalized f32 words to threadgroup storage, then
executes the unchanged metadata Q4 product/accumulation schedule. It preserves
eight sequential square-sum chains, their tree, software divisions, sqrt,
`(x*scale)*rms_weight`, every Q4 rounding boundary and the output tree. No scale
factor is moved through a dot product. Independent groups recompute normalization
instead of using an invalid cross-group barrier.

Prototype domain is K divisible by 256 and <=4096, 64-thread groups. For K=2048,
the normalized tile requests 8192 bytes of dynamic threadgroup memory, in addition
to reduction storage. No weights are expanded. This is more repeated work and
local memory than the standalone producer; the experiment does not establish
which hardware resource caused the measured slowdown.

Twelve real gate/up matrices use the same steps/layers as above. Raw residuals
and RMS weights come from the separate pinned CPU capture. Actual CPU
`Tensor::rms_norm` output is checked against the captured normalized Q4 input,
so the two fixture sets are linked by exact words. Baseline uses the current
standalone RMS `no_contract` kernel followed by metadata Q4. Both fusion modes
use a serial encoder; concurrent dispatch is not combined with this experiment.
Overwrite, residual, guards and timed outputs match CPU exactly.

The three-batch screen gives a candidate/baseline summed GPU-time ratio of
**1.329551503**: **32.9552% slower**. All twelve fixtures regress, ranging from
26.38% to 61.67%. The stop rule triggers. No nine-batch extension, full-model
integration or certificate/recovery campaign is claimed for fusion. Its source
exists only as a GPU test fixture and support harness.

## Validation and retained configuration

Four final release conformance tests pass with zero skips: serial/concurrent
independent Q4 (600 scalar comparisons across modes), protected Q6 scratch reuse
(130 outputs after 64 encoded projections), fused/unfused RMS+Q4 (600 scalar
comparisons), and host metadata round-trip. Guards are checked in addition to
those counts. Synthetic cases cover row tails, widths 256/768/2048/4096, signed
zeros, varied magnitudes, half-subnormal weight scales and residual stores.

The real pilots check 135,168 pre-timing scalar outputs across both concurrent
modes and 270,336 across both fusion modes; final outputs of every measured
batch also match CPU. These are separate from the complete-forward word count.

Additional checks with concurrent dispatch enabled all pass:

- Actual TinyLlama and Stories 2→3→2 regime switches and resets, 768,000 total
  raw logit comparisons against CPU.
- Existing full regime-2 CPU/Metal token/certificate equality.
- Existing regime-3 CPU/Metal token/certificate equality and regime engagement.
- Fresh baseline and candidate codec encodes matching frozen CPU covers, keys,
  distribution tables and full row hashes; actual CPU→concurrent-Metal and
  concurrent-Metal→CPU exact file recovery. The recovery test is one correctness
  trial, not a codec performance campaign.

Retained opt-ins for this optimized research configuration:

```sh
VITNI_METAL_Q4K_VARIANT=metadata
VITNI_METAL_RMS_VARIANT=no_contract
VITNI_METAL_RMS_COOPERATIVE=1
VITNI_METAL_Q6K_COOPERATIVE=1
VITNI_METAL_ATTENTION_COOPERATIVE=1
VITNI_METAL_CONCURRENT_DISPATCH=1
```

Unset or zero concurrent-dispatch flag keeps the previous serial path. The
retained change is scheduling and scratch lifetime protection only. No claim is
made for 7B models, other GPUs/drivers, long contexts or bias-bearing Qwen paths.
The finite tests are not an exhaustive proof against all possible scheduling
hazards. Default/main behavior is unchanged.

## Harness corrections and artifacts

The first complete pair was preserved after an orchestrator error: a result path
still used `cooperative` after its mode label became `candidate`. Only the path
and resume logic changed; pair 0 was not rerun and no observation was discarded.
The initial unoptimized CPU-reference regime-switch test was intentionally
terminated after running slowly; its partial log is saved. All additional checks
then ran to completion in release mode. Neither correction changed measured
candidate binaries or arithmetic. Both are recorded in `driver-corrections.json`.

Parent-workspace artifact:
`outputs/vitni-metal-concurrent-rms-fusion-2026-11-12`. It contains the plan,
original and corrected scripts, all observations/load readings, CPU reference
hash verification, independent probe binaries, immutable baseline/candidate
hashes, exact candidate source snapshot, full histories, fresh ordinary controls,
certificate/regime/codec outputs, partial debug log and final release logs.
The committed summary is `docs/research/concurrent-rms-fusion-2026-11-12.json`.

Reproduce the standalone GPU checks on a real Metal device:

```sh
cargo test --release --manifest-path metal/Cargo.toml --test dispatch_rms_fusion -- --include-ignored --nocapture --test-threads=1
```
