# Cooperative Q6_K hidden projections

November 12, 2026, Apple M3 Max, macOS 27.2.1. Baseline: `bb5495e`, with
Q4 contraction control and cooperative RMS enabled. This experiment changes
regime-2 Q6_K hidden projections; the classifier keeps its current kernel.

## Mechanism and numerical contract

Eight threads per output row compute consecutive 256-value super-blocks
independently. Each block retains the original integer accumulation, activation
quantization, signed scale interpretation and separately rounded float scaling.
The original adjacent-pair floating tree is factored into aligned groups of
eight, then the same tree over their roots. The last partial group carries odd
nodes without adding zeros. This preserves the original tree topology rather
than reassociating floating-point additions.

The topology test checks every block count from 1 through 64. Differential GPU
tests additionally check partial row groups, signed-zero activations, varied
finite scales, residual output and quantization scratch bits against CPU.

The implementation requires 32-thread SIMD execution and 1..64 super-blocks,
matching the existing Metal kernel's `parts[64]` domain. Other widths retain
the existing dispatch. This experiment does not extend or validate the older
Metal backend's wider-matrix capacity. No threadgroup barriers are needed;
each eight-thread subgroup exchanges values through SIMD shuffles.

## Dispatch policy selected from the pilot

Real CPU `LensCapture` checkpoints supply activations at TinyLlama history
steps 0, 7 and 63. Matrices cover Q6 attention values, FFN down projections
and the final output head. The derived Stories15M network supplies a different
768-by-288 Q6 down projection. Every newly captured full CPU output row equals
the earlier frozen reference.

The three-batch pilot compares unchanged quantize-once plus each dot kernel,
including quantization and ordering costs. Cooperative Q6 improves hidden
projections substantially but takes 11–26% more GPU time on the 32000-row head.
That slower path was stopped after the pilot. All complete-model runs keep the
original head; no all-Q6 cooperative promotion run was attempted.

Nine-batch follow-up measurements cover hidden projections only. The kernel
is selected for 20 TinyLlama matrices and two Stories15M matrices. The slower
head results remain in the saved observations; they were not omitted from the
experiment's decision record.

## Complete histories and background contention

Three fresh processes per mode replay the identical 108-token TinyLlama codec
history, positions and resets. The baseline binary is the immutable previously
validated optimized binary; its hash is checked against the previous manifest.
Candidate and baseline both use Q4 `no_contract` and cooperative RMS. Every
complete raw logit row is compared with independent CPU bytes.

| Mode | Median forward time (s) |
| --- | ---: |
| Optimized baseline | 2.115203 |
| Cooperative hidden Q6 | 1.935035 |

This is an 8.518% reduction, with all three paired wins. System load was elevated,
so these values are contention-cohort measurements. The same-cohort ordinary
Metal ratios are 4.150x baseline and 3.796x candidate, not an idle-hardware claim.
The second network also passes all full-bit comparisons; its tiny timing totals
are not used to claim an independent performance result.

A second control alternates baseline/candidate requests token by token, reversing
which runs first on successive steps. Each process keeps its own independent
model state. Both results are checked against CPU on every step. Three fresh
pairs give medians 2.383406s baseline and 2.203485s candidate: 7.549% less forward
time, again three paired wins. This reduces temporal-drift confounding but does
not make the machine idle or replace a lower-load measurement.

The original bounded 180-second observation did not reach a one-minute load
below 7. Those observations remain separate from the subsequent successful
lower-load confirmation below; neither busy cohort is relabeled.

## Lower-load confirmation

A subsequent fresh cohort uses the same immutable optimized-baseline and Q6
candidate binaries. Execution starts after two one-minute-load readings below 7,
separated by ten seconds. All before/after worker samples are between 5.72 and
6.05, below the existing threshold; this is lower load, not a dedicated idle GPU.

Three fresh processes per Vitni mode and three ordinary Metal controls replay
exactly the same frozen 108-token history, positions and resets:

| Mode | Median forward time (s) | Ratio to ordinary |
| --- | ---: | ---: |
| Optimized Q4 + RMS baseline | 2.101172 | 4.092x |
| Cooperative hidden Q6 | 1.937303 | 3.773x |
| Ordinary Metal | 0.513522 | 1.000x |

The candidate reduces forwarding time by 7.799%, winning all three paired runs.
Fresh-process wall time falls from 3.506s to 3.349s. This meets the predeclared
3% minimum gain, three paired wins and no-more-than-2% process regression gates.
All 10,368,000 candidate logit-value comparisons to independent CPU bytes pass;
the baseline passes another 10,368,000. The Q6 configuration reports 20 eligible
hidden matrices; the output head retains the previous kernel.

The cohort qualified on the first attempt. No observation was excluded. These
results confirm the direction and scale of the earlier contended measurements;
use the fresh same-cohort ordinary ratios rather than mixing sessions. One
physical Mac and one complete performance workload remain the scope. Defaults
remain unchanged.

Artifact `vitni-metal-q6-low-load-confirmation-2026-11-12` contains the plan,
immutable-binary provenance, all raw worker traces and load readings, revalidated
comparison, reproduction script and result report. The kernel code was not
changed for this confirmation.

## Recovery and regression checks

Full CPU/Metal certificate agreement and the unchanged regime-3 regression pass.
Fresh baseline/candidate ASCII encodes exactly match the frozen CPU cover,
key, coding tables and complete-row fingerprints. CPU-to-candidate Metal and
candidate-Metal-to-CPU decoding recover the exact original file bytes. Decoders
receive the cover/key without the source payload or encoder trace.

All GPU tests explicitly ran on the GPU. The persistent test covers block counts
1,2,3,7,8,9,22,31,32,33,56,63,64, three seeds and both overwrite/residual forms;
it counts 2460 scalar output comparisons including current-kernel controls,
with quantization and padded-row guards checked separately.

## Trying and reproducing the research path

Before constructing `MetalForward`, set:

```sh
VITNI_METAL_Q4K_VARIANT=no_contract
VITNI_METAL_RMS_COOPERATIVE=1
VITNI_METAL_Q6K_COOPERATIVE=1
```

The new Q6 flag is off by default. Its pipelines are constructed only when
requested. `q6k_cooperative_enabled()` and `q6k_cooperative_matrices()` report
the supported configuration and eligible hidden-matrix count. Regime 3 retains
its existing kernels; the classifier retains its existing kernel in either case.

Artifact `vitni-metal-q6-cooperative-2026-11-12` contains the initial plan, pilot,
dispatch-policy amendment, raw timing observations, captures, immutable binaries,
CPU comparisons, topology/GPU checks, certificate/recovery logs and source hashes.
The initial performance runner stopped at its load preflight before any full
worker. A separate runner retained the busy-cohort results explicitly; no
measurement was deleted or substituted.

One physical Mac/compiler was tested. Other hardware, larger real models,
long contexts and production concurrency remain unvalidated. Default behavior
and the main branch are unchanged. No push, release or paid compute was used.
