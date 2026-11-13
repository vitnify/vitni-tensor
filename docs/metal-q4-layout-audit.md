# Refreshed Metal profile and rejected Q4 layout experiment

November 11, 2026, Apple M3 Max, macOS 27.2.1. Numerical implementation retained:
`a6ae918fb7ea46576d387c14809f6a6ca7240fa0` (cooperative Q4_K and RMS opt-ins).
No further kernel change was accepted and no default changed.

## Lower-load controls

Replay the same 108 input tokens and context resets from the earlier real codec
history. Five fresh processes per timing mode, alternating forward/reverse mode
order. Record system load before and after each worker. All 21 timing/profiling
workers stayed below the predeclared 1-minute-load gate of 7; the maximum sample
was 4.95 on this 14-physical-core host. This is lower load, not a dedicated idle
machine or a guarantee of GPU isolation.

| Mode | Median forward time (s) |
| --- | ---: |
| Cooperative Q4 only | 3.348 |
| Cooperative Q4 + RMS | 2.602 |
| Ordinary Metal, identical input history | 0.494 |

The incremental RMS reduction is 22.27% in this cohort. All observations remain;
none were selected from the earlier busy-host measurements. The combined engine
still takes 5.27 times ordinary forward time on this workload. This is a
matched-token inference comparison, not end-to-end encoding or native parity.

Three further command/dispatch profile pairs preserve every tested output.
Command profiling totals 2.600 s forward and 2.494 s in GPU command execution
intervals. The split-pass diagnostic totals 2.496 s forward; its stage-interval
union is 2.310 s. After merging overlaps within each kernel label, Q4 projections
and accumulating projections cover 0.970 s and 0.458 s, about 62% of that timeline.
RMS covers 0.254 s. Split passes perturb scheduling, so this is diagnostic
attribution, not an exact production breakdown or hardware-utilization counter.
All 1728 Vitni profile/control full-logit-row fingerprints match the CPU-verified
history; the ordinary controls are allowed to differ numerically.

## Candidate and outcome

Test 32-thread groups separately from a Q4_K byte permutation that interleaves
four rows in eight-byte strips. The packed kernel retained every original
dequantization, accumulation chain, chunk boundary and reduction tree. It used
one resident matrix buffer; regime changes unpacked/repacked that buffer rather
than keeping permanent raw and packed copies. The TinyLlama candidate packed
134 matrices (477,167,616 bytes), with no row-padding bytes.

The host byte-layout test passed. CPU-referenced GPU tests passed 3168 scalar
output-bit-pattern comparisons plus row guards; large-shape diagnostic outputs
also matched. Full CPU/Metal certificates and 2→3→2 regime-switch/reset checks
passed. These correctness results did not establish a performance improvement.

Three fresh complete ASCII codec encodes per mode, reversed order on alternating
rounds, gave the following medians:

| Mode | Encoding work (s) | Native forward (s) | Full process (s) |
| --- | ---: | ---: | ---: |
| Existing layout, 64 threads | 3.329 | 2.610 | 7.670 |
| Existing layout, 32 threads | 3.343 | 2.624 | 7.659 |
| Packed layout, 32 threads | 3.515 | 2.800 | 7.759 |
| Ordinary Metal | 0.577 | 0.324 | 4.407 |

Work includes inference, protocol traffic, filtering, table construction and
tracing; it excludes imports/setup/teardown. Ordinary may generate a different
number of tokens. Every Vitni cover, key, table and logit-row fingerprint matches
the frozen CPU reference. All 12 pilot workers completed; no observation was
dropped. Smaller groups change work by +0.42%, and packing by +5.59%.

Neither met the predeclared acceptance rule (at least 5% less median work than
the existing implementation, at least two paired improvements, and at most 2%
cold-wall regression; packing additionally needed to beat the simpler 32-thread
control by 2%). Both were rejected. Candidate source, binary, patch and all
measurements remain in the local artifact `vitni-metal-q4-layout-2026-11-11`.
The active engine was restored to the retained implementation; this document
adds no candidate flag or kernel. Because there was no eligible winner, the
conditional remaining-corpus/recovery promotion trial was not launched.

## Second real-network conformance

The existing pinned [Stories15M Q8_0 checkpoint](https://huggingface.co/ggml-org/models-moved/blob/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M-q8_0.gguf)
was locally requantized through llama.cpp to exercise Q4_K alongside Q5_0, Q6_K
and Q8_0. Its six-layer, width-288 network differs from TinyLlama's topology.
Embedding/output remain Q8_0; the quantizer reports fallback on 36 of 57 tensors.
Four matrices actually use the candidate Q4 layout. This is a derived checkpoint,
not an independently obtained Q4_K release or a model-quality assessment.

Derived GGUF SHA256:
`2deb9657a9e73d84786e2028ee98b71029a02e58f0096f1d2adddd3c53c859b8`.
Candidate 2→3→2 conformance passes 384,000 full-logit value comparisons. The
retained Q4+RMS binary separately passes 768,000 direct output-bit comparisons
against fresh CPU executions in regimes 2 and 3, with a context reset. This
broadens numerical coverage, not performance coverage of larger models.

All tests used the same Mac. CUDA, another physical host, long contexts and
production concurrency remain unvalidated. No push, release, paid resources or
change to the main branch occurred.
