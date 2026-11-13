# Metal resource profile and stopped Q4 lookup pilot

2026-11-12. Research branch: `research/metal-canonical-row-parallel-2026-11-11`.
Numerical baseline: `63f49de` (engine remains at the retained metadata Q4,
RMS `no_contract`, cooperative hidden Q6 and attention configuration). Main
remains `b4d7e6456bef8b7f34c554ca993a41bebe17e317`.

The resource measurements justified trying finished-weight Q4 tables first.
That implementation was exact in the tests but substantially slower in the
kernel screen, so it was stopped before full-model integration. Within-block
Q6 is the next unrun option of these two. No Q6 implementation or speedup is
claimed here, and no active engine or default changed.

## Measurement

Apple M3 Max, 30 GPU cores, 36 GiB. A research predictor wraps the unchanged
engine with programmatic Metal capture and the existing optional profiling
feature. The feature labels pipeline descriptors; it is not the identical
binary used for earlier uninstrumented whole-model timings.

The captured workload is one real TinyLlama Q4_K_M regime-2 forward at context
position 64. It has one command buffer, one encoder and 309 dispatches. Capture
did not enable the split-encoder timestamp mode. Readiness metadata asserts
metadata Q4, RMS `no_contract`, cooperative RMS/Q6/attention and the original
classifier. Model BLAKE3:
`ab7591e1ec49cb5dce27865d31d436091a433a3e807e9920a572c42dda294b0d`.

`gpudebug 1.0` replayed the saved capture twice with overlapping execution:
default state (saved as medium), then fixed high state. Both report consistent
state and eight sampled cores of thirty. Frame GPU durations were 17.2597 and
11.9058 ms. These are debugger replay times, not paired application benchmarks.

Compiler statistics from the default capture:

| Shader | Temp registers | Uniform registers | Spilled bytes | Static instructions | Shader cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 metadata | 29 | 16 | 272 | 317 | 46.50% |
| Q4 metadata + residual | 29 | 16 | 272 | 319 | 17.87% |
| Hidden Q6 cooperative | 60 | 20 | 0 | 1,004 | 1.02% |
| Hidden Q6 cooperative + residual | 60 | 20 | 0 | 1,006 | 4.50% |
| RMS no_contract | 14 | — | 0 | 103 | 21.76% |

Spilled bytes are the compiler's reported quantity, not measured dynamic spill
traffic. Static instruction counts are not executed instruction counts. Shader
cost fractions are the debugger's attribution, not disjoint wall-clock stage
intervals or predicted whole-model gains. The combined Q4 cost is 64.3678%;
hidden Q6 is 5.5231%. The original classifier and quantization are separate.
The high replay's shader-list cost values were unchanged; do not treat that
list as an independent cost-share confirmation. Its runtime counters and
timeline duration did change.

| Whole-device timeline counter | Default | High |
| --- | ---: | ---: |
| External GPU bandwidth | 48.79 GiB/s | 66.04 GiB/s |
| GPU peak bandwidth counter | 6.73% | 5.77% |
| Kernel occupancy | 15.85% | 16.04% |
| Instruction throughput limiter | 32.47% | 29.46% |
| Instruction throughput utilization | 9.79% | 8.85% |
| Integer/conditional limiter | 19.06% | see raw counters |
| F32 limiter | 7.74% | see raw counters |
| L1 cache limiter | 5.45% | see raw counters |

High-state peak-bandwidth samples: mean 5.7704%, p95 12.6673%, maximum
30.3948% across 1,196 samples. Default bandwidth samples: mean 48.7941 GiB/s,
p95 128.1997, maximum 261.3255 across 1,734 samples. Retain Apple's separate
units and normalization; do not infer a physical peak by dividing these values.

These are device timeline counters. Nonzero fragment work confirms other GPU
activity is included. They are not isolated per-shader memory traffic. The
averages do not show sustained bandwidth saturation; they cannot rule out
local memory latency or short peaks. Low occupancy does not identify its cause.

## Stall measurement limitation

Direct per-instruction stall percentages were **not obtained**. The exported
counter tree exposes attempted/executed utilization, launch limiters, occupancy
and memory metrics. A limiter is not a stall percentage, and subtracting a
utilization value from it does not yield a valid stall fraction. The compiler's
`Wait: 0` is a static instruction statistic, not evidence of zero runtime stalls.
All counter descriptions and values are saved, including unsupported/empty
command metrics.

The UI fallback failed: opening installed Xcode 26.6 (17F113) returned
`kLSIncompatibleApplicationVersionErr`, error -10664. The standalone `metal`
compiler also reports a missing downloadable Metal Toolchain; runtime source
compilation and the system command-line GPU debugger work. No toolchain or OS
upgrade, license acceptance or system setting change was performed. Instruments
template discovery worked after allowing its local cache initialization.

Apple describes its [Metal tools](https://developer.apple.com/metal/tools/) and
[counter analysis](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics).
Those capabilities do not establish that every metric was accessible here.

## Choice and bounded pilot

Q4's much larger cost and the available bandwidth evidence favored a small LUT
pilot before spending effort on the smaller hidden-Q6 target. This was a
hypothesis, not evidence that dequantization was the dominant stall source.

The table stores eight groups of sixteen final f32 weight values, followed by
the original 128 nibble bytes: 640 bytes per 256-weight block instead of 192.
Each value is built from the current metadata with a separately rounded
`scale * nibble`, then subtraction. Activation multiplication, eight canonical
lane chains, chunk reduction, tree order and residual addition are unchanged.
The tested layout expands Q4 storage 3.33x; applying it to all of this model's
Q4 matrices would add 1,484,521,472 resident bytes (about 1.38 GiB), excluding
original header retention and construction transients. It was not integrated.

Five pinned CPU-captured matrix/activation fixtures were used, spanning two
real networks. Both modes compile the same contraction-off source, use normal
automatic pipeline thread limits and dispatch eight physical threads per row,
64 per group. Each fixture checks overwrite and residual results plus guards,
warms up, then alternates baseline/candidate order for three batches of sixteen
resident dispatches. GPU command timestamps exclude table construction; its
CPU time and byte cost are recorded separately. Explicit fused/separate
rounding probes validate the contraction setting.

| Fixture (K × M) | Baseline GPU µs | LUT GPU µs | LUT change |
| --- | ---: | ---: | ---: |
| TinyLlama query, 2048 × 2048 | 105.711 | 199.523 | +88.74% |
| TinyLlama key, 2048 × 256 | 65.490 | 91.352 | +39.49% |
| TinyLlama gate, 2048 × 5632 | 116.312 | 115.430 | -0.76% |
| TinyLlama down, 5632 × 2048 | 135.242 | 193.281 | +42.91% |
| Stories down, 768 × 288 | 15.430 | 18.602 | +20.56% |

The sum of fixture medians increased 41.0791%, with four of five losses. This
crossed the prespecified early-stop rule (>5% summed regression and at least
75% losses). This sum is not a model-frequency-weighted whole-forward result.
Table construction also took longer on every fixture. No nine-batch extension,
full-model LUT loading, certificate, codec or regime-switch trial followed.
The apparent 0.76% win is not sufficient evidence to rescue a shape variant.

This is an early screen, not a lower-load confirmation: pre-pilot load was not
recorded because the sandboxed `sysctl` attempt failed. A later escalated sample
was 7.98 / 6.39 / 4.76. Preserve that qualification with the timing; the trial
supports stopping this implementation, not a universal claim against LUTs or a
precise production slowdown. A definitive adoption decision would still require
fresh lower-load whole-model comparisons and setup costs.

## Exactness and scope

- The retained engine through the capture wrapper matched all raw CPU logit
  bytes in the 108-step TinyLlama and 64-step Stories histories: 5,504,000 words,
  including resets and context progression. Captured TinyLlama step 64 matched.
  These are not full-model LUT checks.
- The LUT and baseline each matched the independent CPU matrix outputs in
  overwrite/residual form (20,544 scalar comparisons per mode), plus guards.
  Every timed batch's final overwrite result matched too.
- The archived GPU conformance test checked another 3,156 scalar outputs across
  baseline/candidate modes, plus padded guards. It covers row/group/chunk
  boundaries, signed zeros, cancellation and half-subnormal scale encodings.
  The host metadata round-trip also passed. Two tests passed, zero skipped.
- 191 source-reference files were checked against their earlier saved SHA256
  manifests before the final report. This is finite conformance evidence, not
  an exhaustive proof for all models or float inputs.

The failed LUT is stored only in `metal/tests/fixtures/q4_lookup.metal` and
`metal/tests/q4_lookup.rs`. No change was made to active kernels or host dispatch.
Within-block Q6 has not been built here. Its next pilot should keep quantize-once
and the float block tree, change only bounded integer partial work, and compare
against the current cooperative Q6 kernel. Its small measured cost share gives
no basis to promise elimination of the remaining speed gap.

## Reproduction and artifacts

GPU boundary test (a real GPU is required; ignored tests must explicitly run):

```sh
cargo test --manifest-path metal/Cargo.toml --test q4_lookup -- --include-ignored --nocapture
```

The local artifact directory is
`outputs/vitni-metal-resource-profile-2026-11-12` in the parent workspace. It
contains the predictor source and Cargo lockfile, `run_capture.py`,
`export_profile.py`, `analyze.py`, the full `.gputrace` with both embedded
profiles, counter JSON files and descriptions, sample time series, pilot source,
all timing observations, readiness/model hashes, logs and verified input hashes.
The capture is about 757 MiB and is intentionally not committed.

Command flow used after enabling programmatic capture with
`METAL_CAPTURE_ENABLED=1`:

```text
gpudebug -t <forward.gputrace> --json
profile run --gpu-state default --exec overlapping --embed
profile run --gpu-state high --exec overlapping --embed
info /performance/shaders/comp0 --all
go /performance/timeline/counters
list --all
```

The committed `docs/research/metal-resource-profile-2026-11-12.json` preserves
the numeric summary and the counter definitions. Local `manifest.json` hashes
the source, reference inputs, binaries, exports and capture. Debugger sessions
created for this task were terminated. Nothing was pushed or released.
