# Explicit Metal rounding control and shape specialization

November 12, 2026, Apple M3 Max, macOS 27.2.1. Both experiments preserve the
existing regime-2 arithmetic and were tested separately against the retained
cooperative Q4_K kernel, with cooperative RMS enabled.

## Retained: explicit contraction control

Set `VITNI_METAL_Q4K_VARIANT=no_contract` before constructing `MetalForward`.
This also selects cooperative Q4_K. Set `VITNI_METAL_RMS_COOPERATIVE=1` to
reproduce the measured combination. Defaults are unchanged; `current` selects
the previous source variant. `q4k_variant()` reports the selected variant.

The Q4 cooperative shader uses `#pragma METAL fp contract(off)` and ordinary
float intermediates instead of volatile dequantization/product locals. Fast
math remains disabled. The exact multiply/add boundaries, eight accumulation
chains, low-half-before-high-half order, 8192-element chunks and fixed reduction
trees remain unchanged. Other kernel families and numerical regimes are not
rewritten. An explicit FMA remains available where an algorithm requests it.

Apple documents contraction control separately from safe/precise math in the
[Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf?changes=_6).
The experiment's rounding-sensitive probe produced the separately rounded
answer with the pragma, the fused answer without it, and the fused answer for
an explicit `fma` in either mode. This checks the actual compiler's behavior.
The experiment did not measure generated instructions or register spills, so
the speedup is not attributed to a particular memory or register mechanism.

## Independent experiments

Captured real matrices and activations from the existing TinyLlama and derived
Stories15M runs supplied the microbenchmarks. CPU outputs were the reference.
Tests included overwrite/residual forms, padded-row guards and canonical chunk
boundaries. The first pilot had three alternating batches; the follow-up had
nine, each with 16 resident dispatches. Compilation/allocation were excluded
from these GPU timestamp measurements. Every tested output matched exactly.

The first complete-history cohort used three fresh processes per mode/model:

| TinyLlama mode | Median forward time (s) | Reduction |
| --- | ---: | ---: |
| Current Q4 + RMS | 2.610457 | — |
| Explicit contraction control | 2.110111 | 19.17% |
| Single-chunk/shape specialization | 2.502819 | 4.12% |

All 108 input tokens, positions and resets were fixed, and each full raw logit
row was compared with the independent CPU run. The second network used its
12-token history and reset. All candidate comparisons passed. Background load
was elevated, so the retained candidate received a separate lower-load cohort.
All earlier observations remain in the artifacts.

The shape candidate preserved volatile guards and supplied a dedicated
single-chunk path for widths up to 8192, with compile-time widths 768, 2048 and
5632 and generic fallback above 8192. One pilot shape was slower, so it was not
assumed to be beneficial; its later isolated and full-run results were mixed.
Its 4.12% complete-run gain missed the predeclared 5% retention threshold. It
was archived and removed from the active implementation. It was never combined
with contraction control. No codec promotion campaign was run for it.

## Lower-load confirmation

After removing the shape candidate, rebuild the retained implementation and
run three fresh processes per Vitni mode, plus ordinary Metal on the identical
history. All before/after one-minute load samples were below 7.

| Mode | Median forward time (s) | Ratio to ordinary |
| --- | ---: | ---: |
| Current Q4 + RMS | 2.607910 | 5.043x |
| Contraction control + RMS | 2.109930 | 4.080x |
| Ordinary Metal | 0.517153 | 1.000x |

The retained candidate reduces forward time by 19.095%, winning all three paired
runs. Fresh-process wall time falls from 3.997s to 3.523s. This is an updated
same-cohort comparison; the earlier 5.27x result used a different cohort.
Ordinary logits may differ from Vitni's; both Vitni modes match CPU exactly.

## Complete file recovery and regression checks

Fresh current/candidate ASCII codec encodes produce exactly the frozen CPU
cover, key, coding tables and full-logit-row fingerprints. CPU-to-candidate
Metal and candidate-Metal-to-CPU decoders recover the original bytes exactly;
decoder processes receive only cover/key, not the payload or encoder trace.
Single-observation encoding work is 3.338s current and 2.836s candidate; this is
not a repeated codec timing estimate.

The retained source passes its explicit GPU CPU-reference boundary test over
K=256,768,2048,5632,8192,8448,14336,16640,27648, including partial row groups,
residual accumulation and store guards. Full CPU/Metal certificate agreement
and the existing regime-3 certificate regression also pass. GPU tests were
explicitly executed on a real GPU, not counted as skipped passes.

Across the first cohort and lower-load confirmation, candidate modes pass
33,408,000 direct logit-value comparisons to CPU. Additional codec fingerprints,
isolated matrix checks and certificate comparisons are separate checks.

The artifact `vitni-metal-rounding-shape-2026-11-12` contains plans, all timing
observations, captured-input references, immutable binaries, source hashes,
rejected shape source/patch, reproduction scripts and execution logs. A missing
history-file error in the confirmation runner was corrected before its first
worker ran; the failed preflight log remains alongside the successful run.

Only one physical Mac/compiler was tested. Other GPUs, compiler versions,
larger real models and long contexts remain unvalidated. This is an opt-in
implementation improvement, not native parity or an exhaustive arithmetic proof.
No main-branch edit, push, release or paid compute was used.
