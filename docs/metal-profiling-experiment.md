# Remaining Metal forward cost: November 11, 2026

This experiment profiles the unchanged regime-2 forward after the cooperative
Q4_K optimization. It changes no numerical kernels and does not establish a
further production speedup.

Build the Metal crate with `--features profiling` to enable the research API.
Set `VITNI_METAL_PROFILE=command` before constructing `MetalForward` to collect
whole-command GPU times and host preparation/encoding/wait/readback wall times.
Retrieve a completed step with `take_last_profile()`.

`VITNI_METAL_PROFILE=dispatch` instead creates one compute encoder per dispatch,
with timestamp samples at the encoder boundaries. This mode uses tracked-resource
pass dependencies instead of the production encoder's explicit resource barriers.
It therefore changes scheduling. Compare it against command and off controls;
do not interpret it as an exact decomposition of the production graph. Default
builds do not compile the profiling module. Default execution still uses the
original single-encoder graph.

GPU counter timestamps are calibrated with `MTLDevice.sampleTimestamps` before
and after execution. Its CPU values are already nanoseconds; applying a Mach
timebase conversion again would be wrong. Timestamp resolution, monotonicity and
counter capacity are checked. Unsupported hardware fails explicitly. Raw scalar
buffers are recorded as `u32` words in the `dimensions` field; floating scalar
words are bit patterns, not integer dimensions.

## Measured result

On one M3 Max and TinyLlama 1.1B Chat Q4_K_M, replay all 108 token forwards and
both resets from the earlier real ASCII codec history. Three fresh processes
per mode run serially, with forward/reverse mode order alternated. Keep every
observation, including first-token initialization. Ordinary llama.cpp receives
the same input history and context lengths, rather than generating its own.

| Mode | Median forward time for 108 steps (s) |
| --- | ---: |
| Baseline, profiler off | 5.463 |
| Cooperative Q4_K, profiler off | 3.355 |
| Ordinary Metal, matched input history | 0.503 |
| Cooperative, command profiler | 3.354 |
| Cooperative, split-pass dispatch profiler | 3.257 |

For the cooperative command control, the GPU execution interval totals 3.254 s
of 3.354 s forward time (about 97%). Host preparation totals 0.00062 s, encoding
0.00991 s, and output readback 0.00084 s. These are wall intervals, not hardware
utilization counters. The remaining roughly 0.100 s includes driver, queue and
completion overhead; it is not all attributed to CPU arithmetic.

In the split-pass diagnostic, the union of RMS normalization stage intervals
totals about 1.022 s out of a 3.071 s overall stage union. The kernel performs
each 2048-element row in one GPU thread, including its output loop. This is a
concrete next optimization target. The Q4_K projection and accumulating kernels
cover about 0.968 s and 0.457 s respectively after overlapping intervals of the
same kernel are merged. Other kernel labels can also overlap; these values are
diagnostic intervals, not additive production wall-time shares.

Command profiling changes cooperative forward time by -0.02%; split-pass
profiling changes it by -2.92%. The baseline split mode changes time by -18.01%,
so splitting passes materially changes scheduling and warrants a separate
uninstrumented experiment. No default promotion is justified by this trace.

All 21 histories completed. Across the 18 Vitni processes, every one of 1944
captured full-logit-row SHA256 fingerprints matches the earlier CPU-verified
history. Ordinary output values are allowed to differ; its inputs and context
lengths match. Matching this history measures inference cost, not a fresh
end-to-end encoding/decoding trial. Native speed parity remains unproven: the
cooperative off control is 6.67 times ordinary forward time on this workload.

The full local artifact is named `vitni-metal-profile-2026-11-11`. It contains
the plan, source adapter, frozen history, all observations, raw intervals, merged
occupancy analysis, validation logs and source/binary hashes. The public API
semantics follow Apple's [timestamp conversion documentation](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)
and [counter sampling documentation](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers).
