# Q4 product preparation and gate/up–SwiGLU fusion

November 12, 2026, local date. Apple M3 Max. Research branch
`research/metal-canonical-row-parallel-2026-11-11`; starting commit `d0d5f9b`.
Both candidates are tested separately against the retained concurrent metadata
Q4 / RMS `no_contract` / cooperative hidden Q6 and attention baseline.

**Stop Q4 product preparation:** summed fixture GPU time rises **110.19%**.
**Fusion has a small kernel-chain gain:** **2.58%** in the initial confirmation,
with exact real-model replay. The subsequent
[lower-load full-model confirmation](metal-swiglu-low-load-confirmation.md)
found fusion **2.50% slower**, with losses in all three pairs. Neither candidate
is adopted.
The active engine was restored byte for byte. Main and defaults are unchanged;
there is no push or release and no new remaining-gap ratio.

## Protocol

The frozen plan retains the previous lower-load condition: two one-minute load
averages below 7, at least ten seconds apart, bounded to 180 seconds per wait.
Before/after readings must also remain below 7 for each measured worker. No
background application or system service is stopped to obtain qualification.
This is lower load, not a guarantee of an isolated GPU or fixed GPU clocks.

Three alternating batches of sixteen resident repetitions screen each candidate.
Compilation, packing and allocations are outside GPU timestamps; host encode
and completion intervals are recorded separately. Baseline and candidate use
concurrent encoders, output guards, warmup and matched repetition barriers.
Fusion compares against the entire concurrent gate + up + SiLU-times-up chain.
The baseline Q4 and forward sources are compiled separately as in production.
No samples or first-fixture observations are discarded.

The automatic regression screen is over 5% summed GPU regression with at least
75% fixture losses, or over 10% loss on every primary-model fixture. The user's
instruction to stop a clearly slower candidate also applies. Promising screens
extend to nine batches, then bounded full histories. Adoption still requires
at least 3% median full-history forward gain, three paired wins, at most 2%
fresh-worker regression, and certificate/regime/reset/recovery checks.

All 325 referenced input files were rehashed against earlier pinned manifests.
CPU-captured Q4 outputs and activations remain unchanged. Fusion's CPU epilogue
uses the existing CPU operation sequence and pinned gate/up vectors.

## Q4 product preparation: stopped

One 32-thread SIMD group prepares each row's low-half products concurrently.
The original eight reducers consume products `r, r+8, r+16, r+24` sequentially,
then the high half, then the next sub-block. Scale-by-nibble, subtraction,
activation multiplication, addition chains, chunk boundaries, fixed trees and
residual addition retain their original order. All source lanes execute every
shuffle; invalid rows participate without accessing weights or storing output.
The runner requires actual SIMD width 32 and supported threadgroup limits.
There are two rows per 64-thread group, versus eight in the baseline.

| Captured matrix | Baseline GPU µs | Candidate GPU µs | Time change |
| --- | ---: | ---: | ---: |
| TinyLlama query, K=2048 M=2048 | 64.370 | 225.047 | +249.62% |
| TinyLlama key, K=2048 M=256 | 40.984 | 20.667 | -49.57% |
| TinyLlama gate, K=2048 M=5632 | 83.609 | 238.589 | +185.36% |
| TinyLlama down, K=5632 M=2048 | 139.698 | 235.789 | +68.78% |
| Stories down, K=768 M=288 | 18.797 | 10.219 | -45.64% |

All-five summed median ratio is **2.101863**, with three of five losses. The
automatic all-fixture loss-count threshold alone is not crossed. The primary
TinyLlama sum is **2.190981**, with three of four losses. Stop under the user's
clear-regression instruction: more than double the aggregate GPU cost is not
worth extending. Both fixture wins remain in the evidence; no shape-selective
policy is inferred from this pilot. These sums are screening statistics, not
model-frequency-weighted forward-time estimates.

Real overwrite and residual checks pass **41,088 scalar comparisons including
both modes**, plus guards. Synthetic row/chunk boundary checks pass another
**2,148** comparisons including both modes, plus guards. Tested widths reach
27,648, including 8,192-element boundaries, odd chunk counts, row tails,
signed-zero inputs and subnormal half-scale encodings. Every timed final
output is checked too. No complete-model product-preparation trial was run.
No hardware-counter evidence establishes the cause of its regression.

## Gate/up–SwiGLU fusion

Two separate eight-thread teams compute the paired gate/up output coordinate
using the existing metadata dot helper. Final rounded dot words pass through
eight shared slots and a threadgroup barrier. One epilogue applies the exact
existing `div_sw(g, 1 + vt_expf(-g)) * up` sequence and writes the inner buffer.
The measured kernel has no gate/up debug stores; a separate debug entry point
checks both intermediate vectors. No global inter-workgroup barrier is needed.

The initial pilot is **0.194% slower** in aggregate, with one of six losses.
The prespecified nine-batch confirmation gives **2.5827% less GPU time**, with
all six fixtures winning:

| TinyLlama gate/up pair | Current chain GPU µs | Fused GPU µs | Time change |
| --- | ---: | ---: | ---: |
| Step 0, layer 0 | 171.349 | 168.133 | -1.88% |
| Step 0, layer 11 | 154.234 | 148.534 | -3.70% |
| Step 0, layer 21 | 152.539 | 149.049 | -2.29% |
| Step 7, layer 0 | 151.893 | 148.797 | -2.04% |
| Step 7, layer 11 | 153.206 | 148.687 | -2.95% |
| Step 7, layer 21 | 152.393 | 148.250 | -2.72% |

Each campaign's pre-timing real checks compare **135,168 final inner words**
and **135,168 intermediate gate/up words**, including both modes and debug /
measured variants as applicable. Timed final words and guard regions also pass.
Synthetic finite-exp comparisons cover **1,092 intermediate** and **819 final**
words, including controls and both candidate entry points, plus guards.

The temporary full integration accepts matching regime-2 metadata-Q4 pairs,
with type, layout, shape and pipeline checks. Other cases retain the original
chain. It keeps the old gate/up allocations for fallback; no persistent-memory
saving is claimed. It adds an optional pipeline but no expanded weight format.
Readiness reports eligible layers, and every step reports actual fused calls.
On TinyLlama it executes 22 fused layer calls per step. The Stories model has
no eligible gate/up pairs and executes zero fused calls; that is fallback
coverage, not a second fused-model performance result.

### Initial complete replay and timing limitation

Four 180-second lower-load preflights expired without starting any timing
worker. Load initially rose to 19.15, declined without reaching the threshold,
and later rebounded to 21.90. Every reading and failed preflight is preserved. The
threshold was not relaxed. The lower-load pilots above qualified independently.

A separate, explicitly numerical-only run replays the pinned 108-step TinyLlama
history and its resets, then the 64-step Stories history. Candidate and baseline
each match **5,504,000 raw CPU logit words**: **3,456,000** with fusion actually
engaged on TinyLlama, and **2,048,000** with fallback on Stories. Before/after
one-minute loads in this numerical cohort are approximately 14–15. Its elapsed
times are retained only as raw observations and excluded from performance,
break-even and adoption claims.

This initial trial established no full-model speedup, setup-cost improvement,
new ordinary-Metal ratio or reduction of the retained approximately 2.47x gap.
The [subsequent lower-load confirmation](metal-swiglu-low-load-confirmation.md)
resolved the pending measurement: fusion is 2.50% slower in full-history
forwarding. Fusion remains archived. The certificate/regime/bidirectional
recovery promotion campaign is not run because the performance gate failed.

### Existing baseline overflow limitation

The first unscaled synthetic fusion check found a baseline-versus-CPU mismatch
before testing the candidate: sufficiently negative gate values overflow the
SiLU exponential, and existing Metal software division returns a different
result from CPU. With gate approximately -28,236.8 and up 251,402.88, CPU inner
bits are `80000000` (negative zero); baseline and fused Metal give `7f800000`
(positive infinity). Another case at gate approximately -3,982,087.5 reproduces
the problem. Gate and up themselves match CPU bit for bit in both cases.

The committed regression makes this limitation visible and verifies that both
fused entry points preserve baseline behavior. Separate CPU comparisons scale
synthetic inputs by powers of two to bound gate magnitudes by 16; real captures
and histories are never scaled or filtered. Initial failing logs remain saved.
This experiment does not repair the existing overflow issue or establish
universal IEEE-754 equivalence. No such mismatch occurred in the real captures
or complete histories tested here.

## Reproduction and archive

Standalone research tests from the repository, on a real Metal GPU:

```sh
cargo test --release --offline --manifest-path metal/Cargo.toml \
  --test q4_prepare_swiglu -- --include-ignored --nocapture --test-threads=1
```

The workspace artifact `outputs/vitni-metal-q4-prepare-swiglu-2026-11-12`
contains the plan, verified input hashes, all pilot/confirmation batches,
immutable baseline/candidate/test binaries, numerical histories, load logs,
failed preflights, original and final test sources, and the complete fusion
integration patch. The committed patch is
`docs/research/gate-up-swiglu-trial.patch`; it is inactive. All four final tests
pass, with no ignored or filtered-out tests. The standalone tests and shader
fixtures remain runnable without applying that full integration.

The original `run_complete.py` and saved binary remain archived. The follow-up
report links the executed confirmation driver, using the existing benchmark
virtual environment for ordinary NumPy/llama.cpp controls. That cohort preserves
all observations and excludes the initial high-load numerical-only runs. Neither
candidate was combined with RMS staging, within-block Q6 or any other archived
optimization. One Mac/compiler and bounded numerical fixtures do not validate
other physical GPUs, larger fused models or long contexts.
