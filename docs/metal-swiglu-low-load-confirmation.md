# Lower-load full-model gate/up–SwiGLU confirmation

November 12, 2026 local date. Research branch `research/metal-canonical-row-parallel-2026-11-11`,
starting commit `492ee61`. Apple M3 Max.

**Reject the fusion candidate:** median full-history forward time is
**2.5027% slower**, with losses on all three
pairs. Every tested raw logit word still matches CPU. The active concurrent
baseline remains unchanged. No main/default change, push, release or paid
compute occurred.

## Matched full-model result

TinyLlama 1.1B Chat Q4_K_M, the pinned 108-token history and its original
positions/resets, context 64, regime 2. Each observation comes from a fresh
native process; pair order alternates baseline/candidate, candidate/baseline,
baseline/candidate. Both modes assert the existing concurrent metadata Q4,
RMS `no_contract`, cooperative hidden Q6 and attention configuration, with the
original classifier. Only the candidate enables gate/up–SwiGLU fusion. It
asserts 22 eligible layers and 22 actual fused calls on every TinyLlama step.
No other archived optimization is combined with it.

| Pair | Concurrent baseline forward s | Fused forward s | Candidate change |
| --- | ---: | ---: | ---: |
| 0 | 1.260897003 | 1.299574540 | +3.07% |
| 1 | 1.256647286 | 1.288096839 | +2.50% |
| 2 | 1.243500960 | 1.281813790 | +3.08% |
| Median | **1.256647286** | **1.288096839** | **+2.50%** |

Median setup is 0.611942209 s baseline versus
0.629908708 s candidate. Median fresh-worker time,
including setup, protocol, sorting and CPU-word verification, is
2.778586292 versus 2.826445292 s:
**+1.7224%**. Preserve the first workers and every
observation; no warm-only subset or outlier trimming is used.

The first pair's forward regression is 3.07%,
below the prespecified immediate stop thresholds of over 10% forward or over
15% worker regression. The bounded three-pair confirmation therefore completes.
All three losses reject the candidate; there is no further fusion performance
sweep or certificate/regime/codec promotion campaign. The adoption gate required
at least 3% median forward improvement, three paired wins and no more than 2%
worker regression; its forward conditions fail.

## Lower-load qualification and ordinary controls

Before every measured worker, require two consecutive one-minute load averages
below 7, at least ten seconds apart, with a bounded 180-second preflight. Each
worker also requires before/after values below 7. All eleven workers qualify
(six TinyLlama, three ordinary controls, two second-model numerical controls).
Observed one-minute readings span **3.93–6.72**. Five- and
fifteen-minute averages remain elevated and are preserved. No application or
system service is stopped or reconfigured. This is the stated lower-load
condition, not idle hardware or guaranteed absence of other GPU work.

Fresh ordinary-Metal controls replay the same history. Forward observations are
0.510426452, 0.511861457, 0.506945916 s;
median **0.510426452 s**. Same-cohort forward ratios are
**2.46196x** concurrent baseline and
**2.52357x** fusion. Ordinary arithmetic may differ.
The candidate is rejected, so the retained approximately 2.47x baseline is not
replaced with the candidate ratio or pooled with an earlier denominator.

The earlier 2.58% isolated gate/up/SiLU-chain gain did not translate into a
whole-model gain. This cohort does not identify a hardware-counter cause.

## Exact replay and scope

Baseline and candidate each compare **12,416,000 raw
logit words** directly to independent pinned CPU files: three 108-step TinyLlama
histories (10,368,000 words per mode), plus a 64-step Stories history (2,048,000).
Every input position, context reset, model identity and numerical configuration
is checked; comparison uses exact bytes, not tolerances or selected tokens.

Fusion actually runs only on TinyLlama. Stories has no eligible metadata-Q4
gate/up pairs and confirms zero fused calls on every step; its agreement is
fallback coverage, not a second fused-model speed result. No existing high-load
numerical-only run or earlier pilot timing is pooled into this cohort.

This is the same immutable candidate binary used in the earlier numerical
proof, SHA256 `dd055779ab6cda7313005055602d7b9dcf2e014aab1c9425289b66c8734e875d`. The baseline binary is SHA256
`2bac0837d5280e5874cb8e2b4409fc645ec90e75e282f9c8093d0bfff7098a0b`. All 325 referenced input files, both binaries and
the unchanged engine source were verified before execution. No new kernel,
compiler flags or active runtime path was introduced for confirmation.

The existing baseline SiLU exponential-overflow limitation remains documented
in [the original experiment](metal-q4-prepare-swiglu-experiments.md). It is not
fixed by fusion, and this finite-model replay does not establish universal
IEEE-754 equivalence. Other physical GPUs, larger fused models and long
contexts remain unvalidated. No additional candidate certificate, regime-switch
or bidirectional file-recovery campaign is claimed.

## Reproduction and archive

The inactive prototype remains in `docs/research/gate-up-swiglu-trial.patch`
and the standalone test fixtures from the original experiment. It is not
enabled in the active engine. The workspace artifact
`outputs/vitni-metal-swiglu-low-load-confirmation-2026-11-12` preserves the
frozen plan, source/input/binary hashes, confirmation driver, every native
worker result, readiness and per-step call counts, ordinary controls, complete
load observations, comparison, source checks and manifest.

The driver reuses saved immutable binaries and the existing benchmark virtual
environment; it refuses to overwrite observations. From the workspace root,
the command used was:

```sh
work/vitni-subtext-gonka-venv/bin/python \
  outputs/vitni-metal-swiglu-low-load-confirmation-2026-11-12/run_confirmation.py
```
