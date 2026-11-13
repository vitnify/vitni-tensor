# Cooperative attention with the existing numerical profile

November 12, 2026, Apple M3 Max. Baseline: immutable `f1fbac0` code with
Q4 `no_contract`, cooperative RMS and cooperative hidden Q6 enabled. The
classifier retains its original Q6 kernel in both modes.

## Mechanism

One 64-thread group handles each query head. Threads independently compute
query-key scores at different positions, exponentials and attention-value
output columns. Every dot preserves the original eight accumulation chains,
element order, separately rounded products and fixed tree. The maximum scan
remains serial and ordered. Eight threads compute the original denominator
chains; thread 0 combines them through the original tree and software divide.
Shared scores are normalized before the original canonical A.V dots.

The kernel uses explicit contraction control for its multiplication/addition
boundaries. Existing `vt_expf`, `div_sw`, square root and grouped-query mapping
are retained. It does not substitute an ordinary FlashAttention reduction,
approximate exponential or different softmax profile.

The existing domain remains context lengths 1..512 and head sizes 1..8192.
The launcher uses exactly 64 threads per query head. Unsupported configured
head sizes or threadgroup capacity keep the previous pipeline. No broader
context support or numerical-regime change is introduced.

## Real inputs and isolated tests

CPU `LensCapture` checkpoints supply actual attention outputs before the output
projection. Query vectors are reconstructed with the same RMS, projection,
bias and RoPE arithmetic; KV data comes from the actual post-RoPE CPU cache.
An independent attention calculation using public CPU canonical-dot and softmax
operations must match the real forward checkpoint before inputs are saved.

Capture steps 0,7,31,63 and first/middle/last layers. TinyLlama uses its frozen
108-token codec history; all newly captured CPU logits equal the prior frozen
reference. Derived Stories15M uses a fixed 64-token history to reach the same
context lengths, with its original 12-token sequence repeated. This extends
numerical coverage; it is not a model-quality or text-generation benchmark.
TinyLlama has 32 query heads, head size 64 and GQA factor 8. Stories15M has six
heads, head size 48 and GQA factor 1.

All 24 captured cases improve in the three-batch pilot, including context 1;
no slower-context fallback or early-stop rule is triggered. Nine alternating
batches of 16 resident dispatches follow. Both pipelines exactly match CPU
outputs and padded output guards. Compilation and allocation are excluded from
these isolated timings; all raw observations remain in the artifacts.

The persistent GPU differential test passes 343,652 scalar comparisons including
current-kernel controls, plus output guards. It covers GQA, signed-zero inputs,
head sizes 1,7,31,48,64,65,128 and selected 8192-head-size boundary cases, and
contexts 1,2,7,8,9,31,32,63,64,65,127,128,511,512 with three seeds.

## Complete replay and timing

Three fresh serial processes per mode replay the same TinyLlama 108-token input
history, positions and resets; three ordinary Metal controls receive identical
inputs. A second 64-token real-network history is checked separately. Every
complete Vitni logit row is compared directly to independent CPU bytes.

| TinyLlama mode | Median forward time (s) |
| --- | ---: |
| Optimized Q4/RMS/Q6 baseline | 1.947402 |
| Cooperative attention | 1.677553 |

This is 13.857% less forwarding time, with three paired wins. The same-cohort
ordinary ratios are 3.720x baseline and 3.204x candidate. The second-network
medians are 0.369206s and 0.327249s, with complete bit equality. System load
exceeds the lower-load gate, so these timing estimates are provisional.

A separate control alternates baseline/candidate requests token by token and
reverses which runs first. The independent processes keep separate model states;
both outputs must equal CPU after every step. Three fresh pairs yield medians
1.959547s baseline and 1.683994s candidate: 14.062% less forward time and three
paired wins. Both models are resident, so these totals are not pooled with the
single-worker cohort. Background load remains elevated in this control.

The original bounded 180-second lower-load attempt did not qualify; its
samples ranged from 24.30 to 40.19 and the estimates were recorded as provisional.
Those observations remain separate from the subsequent successful lower-load
confirmation below. No background app or system process is changed.

## Lower-load confirmation

A subsequent cohort uses the same immutable attention candidate and optimized
Q4/RMS/Q6 baseline binaries. Two one-minute-load readings below 7, ten seconds
apart, precede execution. All before/after worker samples lie between 5.94 and
6.41. This satisfies the existing lower-load gate, without claiming an idle GPU.

Three fresh serial processes per Vitni mode and three ordinary Metal controls
replay the identical frozen 108-token history, positions and resets:

| Mode | Median forward time (s) | Ratio to ordinary |
| --- | ---: | ---: |
| Optimized Q4/RMS/Q6 baseline | 1.940453 | 3.823x |
| Cooperative attention | 1.671307 | 3.292x |
| Ordinary Metal | 0.507632 | 1.000x |

Forward time decreases 13.870%, with all three paired wins. Fresh worker wall
time, including setup/protocol/verification, falls from 3.327s to 3.045s. The
3% minimum forward gain, three paired wins and no-more-than-2% worker regression
gates pass. The first cohort qualifies; no observation is excluded.

Every complete raw Vitni logit row matches independent CPU bytes. The candidate
passes 10,368,000 new logit-value comparisons, with the baseline passing another
10,368,000. Q4 contraction control, cooperative RMS and cooperative hidden Q6
remain enabled in both modes; the original classifier remains selected. The
kernel code, numerical profile and defaults are unchanged for this follow-up.

This confirms the earlier approximately 14% direction and scale under lower
load. Use the new same-cohort ordinary ratios; do not combine observations or
controls from different sessions. One Mac and this workload remain the scope.
Artifact `vitni-metal-attention-low-load-confirmation-2026-11-12` preserves the
plan, binary/source provenance, load readings, every worker result, reproduction
script and revalidated report. The implementation stays opt-in on research.

## Recovery and regressions

The retained candidate passes full CPU/Metal certificate equality and the
existing regime-3 certificate regression. Fresh baseline and candidate ASCII
encodes reproduce the frozen CPU cover, key, coding tables and complete-row
fingerprints. CPU-to-candidate Metal and candidate-Metal-to-CPU decoders recover
the exact original bytes without receiving the source payload or encoder trace.

The complete-history and interleaved cohorts pass 26,880,000 candidate logit-value
comparisons to CPU. Isolated attention comparisons, guards, certificate checks
and codec fingerprints are additional checks. This is observed agreement, not
an exhaustive proof over every IEEE-754 input.

## Trying the research implementation

Before constructing `MetalForward`, enable the existing optimized baseline and
the new opt-in:

```sh
VITNI_METAL_Q4K_VARIANT=no_contract
VITNI_METAL_RMS_COOPERATIVE=1
VITNI_METAL_Q6K_COOPERATIVE=1
VITNI_METAL_ATTENTION_COOPERATIVE=1
```

The attention flag is off by default. Its pipeline is constructed only when
requested. `attention_cooperative_enabled()` reports the supported configuration.
The shared attention operation is used in both numerical regimes, and both
certificate checks explicitly run with this flag enabled.

Artifact `vitni-metal-attention-2026-11-12` preserves the plan, real inputs and CPU
reference outputs, pilot/follow-up timings, complete and interleaved histories,
load observations, immutable binaries, recovery checks, source hashes and scripts.
Only one physical Mac/compiler is tested. Wider real models, long complete
contexts, other GPUs and production concurrency remain unvalidated. Defaults
and main remain unchanged; no push, release or paid compute was used.
