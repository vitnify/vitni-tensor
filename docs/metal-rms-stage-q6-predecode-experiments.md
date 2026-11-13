# RMS product staging and persistent Q6 predecode

2026-11-12 local date. Research branch
`research/metal-canonical-row-parallel-2026-11-11`, baseline `7fefabc`.
Both candidates are compared independently to the retained concurrent baseline:
metadata Q4, RMS `no_contract`, cooperative hidden Q6/attention, concurrent
dispatch and the original classifier. Neither candidate is adopted.

RMS staging preserves exact outputs and improves full-history median forwarding
by **2.1957%**, below the preset 3% gate. Q6 predecoding preserves exact kernel
results but is **15.7560% slower** in its real-input screen, triggering early
stop. Active host/kernel source was restored byte for byte to the concurrent
baseline. Defaults/main are unchanged; nothing was pushed or released.

## Design and correctness contracts

RMS staging lets all 64 threads produce separately rounded `x*x` values into
dynamic threadgroup storage. After a new barrier, the original eight reducers
read their exact original `i%8` sequences. The addition chains, tree, software
division, sqrt, epsilon and `(x*scale)*weight` order remain unchanged. A row is
normalized once. This is separate from the previously rejected RMS-to-Q4 fusion.

The prototype supports widths 1..4096 with a capacity check and conservatively
reserved reduction space. Width 2048 uses 8192 bytes of square storage plus
small static reduction state. Other widths select the old implementation. The
full prototype clears the dynamic-memory binding after dispatch so subsequent
kernels do not inherit the allocation. It adds no persistent model buffers.
Local contraction is disabled as in the retained RMS kernel.

Persistent Q6 predecode stores each 256-weight block as 256 signed int8 values,
16 original signed scale bytes, two original half-header bytes and two zero
padding bytes: **276 versus 210 bytes**, a 31.4286% affected-buffer increase.
All integer weights remain in -32..31; no quantization is changed. The original
four interleaved integer accumulators, scale accumulation, per-block f32 products,
eight-thread cooperation and float tree remain. Quantize-once is unchanged.
Only hidden projections are screened; the original classifier policy is kept.

The raw low/high bit planes can be reconstructed from the decoded integers;
scales and original half bits are copied back. A host test round-trips arbitrary
quant bits/scales and **all 65,536 half headers**, including sign, subnormals,
infinities and NaN payload encodings, without float conversion. This is byte
restoration evidence, not a claim that nonfinite models are valid inference
inputs. Valid integer block sums remain bounded by 133,169,152, below int32 max.

## Protocol and load qualification

Use local Apple M3 Max. Require two consecutive one-minute load averages below
7 at least ten seconds apart before each cohort, pair or ordinary worker; wait
at most 180 seconds. Record before/after loads. Never close background apps to
force qualification. All measured workers qualified.

Load was initially 7.05, and the driver waited for 6.87 and 6.81 before the RMS
pilot. Later load rose above the gate before the third complete pair; the driver
waited again and only started after two qualifying readings. Five/fifteen-minute
averages remained elevated. This is the declared lower-load condition, not idle
hardware or proof of absent GPU contention. Every wait and reading is retained.

Prespecified kernel screen: three alternating batches of sixteen resident
repetitions. Stop any numerical mismatch, or >5% summed GPU regression with at
least 75% fixture losses (also stop >10% regression on all primary-model inputs).
Otherwise extend to nine batches and a bounded full-history comparison.
Adoption requires >=3% median forward gain, three paired wins, <=2% median
fresh-worker regression and subsequent certificate/regime/recovery checks.
First-pair complete-history regression stops are >10% forward or >15% worker.

All inputs were verified against their earlier SHA256 manifests. Both probe
encoders use concurrent dispatch. Kernel compilation, packing and allocations
are outside the GPU timers; the full RMS worker includes setup and dispatch
binding costs. Kernel sums are not weighted whole-model savings.

## RMS results

Twenty-four real captured norm inputs span TinyLlama/Stories, steps 0 and 7,
first/middle/last layers and both attention/FFN norms. Current `no_contract`
and staged products match the independent CPU RMS outputs in out-of-place and
in-place stores, including guards. Every measured batch's final output is also
checked.

The pilot reduces summed GPU time **17.6449%** with 24/24 wins. Nine-batch
confirmation reduces it **18.0301%**, again 24/24 wins. This qualified the RMS
candidate for the bounded complete-history comparison.

Three fresh native-process TinyLlama pairs alternate order. Each replays the
pinned 108-token history with its reset, context 64 and regime 2. Both modes
assert concurrent dispatch. Candidate readiness additionally asserts actual
`product_stage` selection; baseline is an immutable retained concurrent binary.

| Pair | Concurrent baseline forward s | RMS-staged forward s |
| --- | ---: | ---: |
| 0 | 1.256617000 | 1.234943005 |
| 1 | 1.262618624 | 1.226146463 |
| 2 | 1.242607261 | 1.229026002 |
| Median | **1.256617000** | **1.229026002** |

Median forward gain: **2.1957%**, three paired wins. Median fresh-worker time
falls from 2.801796917 to 2.724286292 seconds, **2.7665%**. Median setup is
0.607833792 versus 0.614781625 seconds. The first candidate worker is slower
overall (2.898202500 versus 2.853049583), including setup 0.763426834 versus
0.642564042. These observations remain in the result.

Fresh ordinary-Metal control processes give 0.531490462, 0.511721502 and
0.510420497 seconds, median 0.511721502. The same-cohort ratios are 2.45567x for
the concurrent baseline and 2.40175x for the staged trial. The latter is an
archived candidate result, not a new retained baseline or a denominator to pool
with earlier runs.

A complete 64-step second-network history also matches CPU. Across three
TinyLlama candidate runs and that Stories run, **12,416,000 raw logit words**
match independent CPU files. Baseline controls independently match the same
count. The second-network pair is numerical validation, not a speed claim.

RMS boundary tests compare **794,916 scalar output words** across both modes,
in/out-of-place stores, widths 1..4096, row/tail cases, signed-zero and varied
magnitudes, three epsilon values and guards. They pass with a real GPU, zero
skips. This is finite conformance evidence. The full-model gain misses the
predeclared 3% gate, so no certificate/regime/codec promotion campaign followed.
No extra timing cohort was run to seek a passing result.

## Q6 results: early stop

Eight real hidden projections cover TinyLlama value/down projections at steps
0, 7 and 63 plus Stories down projections at steps 0 and 7. Raw and expanded
buffers are prepared once per fixture and remain resident through alternating
timed batches. Timers include unchanged quantization and the cooperative dot;
packing time and byte counts are recorded separately. Both modes use eight
threads per row, 64-thread groups, and the concurrent scratch-reuse barriers.

| Fixture | Baseline GPU µs | Predecoded GPU µs | Change |
| --- | ---: | ---: | ---: |
| Tiny step 0 value | 68.057 | 61.992 | -8.91% |
| Tiny step 0 down | 102.331 | 139.312 | +36.14% |
| Tiny step 7 value | 26.708 | 26.117 | -2.21% |
| Tiny step 7 down | 90.219 | 106.672 | +18.24% |
| Tiny step 63 value | 23.740 | 26.091 | +9.91% |
| Tiny step 63 down | 83.138 | 99.971 | +20.25% |
| Stories step 0 down | 21.706 | 23.279 | +7.25% |
| Stories step 7 down | 19.766 | 20.872 | +5.60% |

Summed fixture GPU time increases **15.7560%**; six of eight lose. The stop rule
triggers. The two small value-projection wins do not justify a shape-specific
rescue. No nine-batch extension or full-model Q6 integration was attempted.
The experiment does not establish a causal register/bandwidth explanation.

CPU-matching real overwrite/residual outputs total 14,976 scalar words per
mode, plus guards and timed final outputs. Quantization scales and all int8
activations also match CPU. GPU boundary/extreme tests compare another 3,216
scalar words across modes, plus quantization and guards. The float-tree topology
is checked for block counts 1..64. Four Q6 tests pass, zero skipped, including
the exhaustive half-header byte-roundtrip test. There is no full-model Q6 replay
claim, nor a measured whole-model setup/peak-RSS claim for the expanded layout.

## Archive and reproduction

Active source is restored to the exact baseline. The full RMS integration is
archived in `docs/research/rms-product-stage-trial.patch`, checked with
`git apply --check` against the restored branch. It contains the opt-in host
path, dynamic-memory binding and kernel; it is not an adopted optimization.
The standalone GPU fixtures remain under `metal/tests/fixtures/` and the
lossless Q6 layout under `metal/tests/support/q6_predecoded.rs`.

```sh
cargo test --release --manifest-path metal/Cargo.toml --test rms_product_stage --test q6_predecoded -- --include-ignored --nocapture --test-threads=1
```

Parent-workspace artifact:
`outputs/vitni-metal-rms-stage-q6-predecode-2026-11-12`. It contains the plan,
immutable binaries, source snapshots, all batch/full-history observations,
all load waits, CPU reference verification, pack costs/byte counts, fresh
ordinary controls, raw-word hashes, tests and full integration patch. The
committed numeric record is `docs/research/rms-stage-q6-predecode-2026-11-12.json`.
Neither result validates other hardware or larger models, and no combination
of the candidates was built or measured.
