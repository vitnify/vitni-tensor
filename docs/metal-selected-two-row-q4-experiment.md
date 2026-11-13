# Shape-restricted two-row Q4: exact replay, small full-model gain

November 12, 2026. Test only metadata Q4 matrices with **K=2048, M=5632**.
The optimized baseline enables metadata Q4, RMS local contraction control,
cooperative Q6/attention and the original classifier. Keep ordinary automatic
pipeline thread limits and 64-thread dispatch groups. Other shapes retain the
baseline kernel; the previously rejected global two-row policy is not revived.

The selective trial preserves the archived shader's per-row rounded operations,
eight canonical chains, chunk boundaries, reduction trees and residual addition.

## Independent shape confirmation

Use fresh CPU forwards through steps 0–7 and assert all eight full logit rows
against pinned original CPU bytes. Capture both gate/up projection inputs,
weights and CPU output rows at steps 0 and 7, first/middle/last layers: twelve
actual input/weight combinations. All have the preselected shape; no new winner
or dispatch threshold is chosen after observing their timings.

Three alternating batches of sixteen resident dispatches per variant/input pass
the screen with 5.99% less summed GPU time. A nine-batch follow-up gives 10.67%
less summed GPU time; all twelve inputs are faster. Every tested overwrite,
residual and guarded output matches the actual CPU projection bit-for-bit.
Allocation/compilation are excluded from these warm resident-kernel timings.
The arithmetic shader is identical to the earlier boundary-tested prototype;
this trial does not modify its addition order or weight representation.

## Full-model trial

The temporary engine option selects two-row tiling only when all of these hold:
regime 2, metadata Q4, supported 64-thread pipelines, K=2048 and M=5632.
All other paths retain the previous kernel. Instrument actual encoded dispatches
and reset the count at each forward. Assert **44 selected dispatches per TinyLlama
step** and **zero** on the derived Stories15M network, which has no eligible
matrices. The trial's counter overhead is included in timings.

Replay the frozen 108-token TinyLlama codec input history with original positions
and two resets, context 64. Three fresh serial workers per Vitni mode, reversing
order on alternating repetitions, plus three fresh ordinary Metal controls with
identical inputs. Every raw logit byte is compared directly to independent CPU.
The two-sample lower-load start and before/after load gates pass. No background
application is changed and no timing observation is discarded.

| Median seconds | Optimized baseline | Selective two-row trial |
| --- | ---: | ---: |
| Forwarding | 1.445332 | 1.427376 |
| Fresh worker, including setup/protocol/verification | 2.915130 | 2.920444 |
| Setup | 0.560133 | 0.576668 |

Forwarding improves **1.2424%**, with three paired wins. Fresh-worker time rises
0.1823%, effectively unchanged in this small trial. The first candidate worker
is slower: 3.316665 s versus 3.030204 s, including 1.009792 s setup; it remains
in the samples. These are fresh processes with normal system caches, not forced
cache-cold measurements.

Ordinary Metal forwarding is 0.508773 s. Same-cohort ratios are **2.8408×** for
the baseline and **2.8055×** for the trial. These are specific to this cohort;
they do not replace earlier denominators in cross-session improvement claims.

The shape-level gain therefore does not transfer strongly to whole forwarding.
Only 44 projection dispatches per step change, and the resident microbenchmark
does not reproduce complete-forward scheduling/cache conditions. No hardware
counters establish a more specific causal explanation.

## Exactness and decision

All **12,416,000 candidate logit values** match independent CPU raw bytes:
three 108-step TinyLlama histories and one 64-step derived Stories15M history.
Baseline controls pass another 12,416,000 values. Every saved full-row hash
revalidates against the pinned CPU files; no tolerance or top-token-only check.
The counters verify the intended shape selection on every measured step.
Stories15M checks the unchanged fallback, not performance of the selected kernel.

The trial misses its **predeclared 3% median forward improvement threshold**.
It passes the three-paired-win and at-most-2% worker-regression conditions, but
these do not override the missed forwarding gate. Do not lower the threshold
after observing the result. The first-pair early-stop condition was not reached;
the bounded full timing trial completed normally.

**Archive the selective implementation and retain the prior active baseline.**
The research patch is [selected-two-row-q4-trial.patch](research/selected-two-row-q4-trial.patch).
Its applicability to the restored production source is checked. No selective
runtime option, pipeline, counter or new active shader remains in the engine.
The earlier global experiment's ignored GPU test/fixture remain unchanged.
Main and default behavior are unchanged.

Because there is no retention-eligible winner, no new certificate, regime-switch
or cross-backend codec campaign is launched. Exactness claims cover the full
tested forward histories; no fresh file-recovery claim is made for this trial.
The existing active baseline retains its prior certificate/recovery evidence.

## Reproduction and scope

The workspace artifact `outputs/vitni-metal-q4-two-row-selected-2026-11-12`
contains the predeclared plan, fresh captures, both kernel timing phases, all
per-step dispatch counts and CPU-checked rows, ordinary controls, immutable
binaries, trial source snapshot/patch, command/build logs and hashes.

The trial predictor binary is self-contained. Rebuilding its source requires
applying the archived patch to the research checkout first; the trial adapter
uses the temporary instrumentation APIs that were removed from active source.
That option was `VITNI_METAL_Q4K_TWO_ROW_SHAPE=1`, together with metadata Q4,
RMS `no_contract`, cooperative Q6 and cooperative attention. It is not an option
in the restored engine. Offline compilation uses the existing Cargo cache;
Metal execution requires host GPU access.

One physical M3 Max/compiler, one complete performance history and twelve warm
shape fixtures do not validate other hardware, larger models, long contexts,
production concurrency or native parity. No novelty claim, paid compute, main
change, push, release or background-application mutation occurred.
