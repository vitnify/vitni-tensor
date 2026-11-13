# Two-row Q4 register tiling: stopped after the pilot

November 12, 2026, research branch. The retained baseline is `d401c27`:
metadata Q4, RMS local contraction control, cooperative Q6/attention and the
original classifier. This experiment isolates its Q4 metadata kernel. Other
forward stages are not executed by the kernel pilot.

## Mechanism and exactness

Each eight-thread subgroup computes two independent output rows instead of
one; a 64-thread group therefore handles sixteen rows. Keep the existing
192-byte metadata blocks and ordinary automatic pipeline thread limits. Load
each activation once for the pair, prepare the two rows' own products and
update separate canonical accumulators. Every row retains its original element
sequence, separately rounded multiply/subtract operations, low-before-high
ordering, 8192-element chunk boundaries, adjacent-pair trees and residual add.
Contraction remains explicitly disabled; no hardware sum or reassociation.

Two sets of shared partials replace one: 512 bytes per group instead of 256.
Each lane also carries two sums and two chunk-root arrays. These source-level
facts are not measurements of register count, occupancy or spills. The missing
partner of an odd final row never reads an out-of-range weight block or stores
an output. All padded groups still participate in the barriers.

The CPU-reference boundary test passes **3,156 scalar comparisons including
both baseline and candidate controls**, plus output guards. Cases cover row
counts 1, 2, 7, 8, 9, 15, 16, 17, 31, 33 and 65; widths 256 through 27648;
chunk transitions and an odd number of chunks; three seeds; signed-zero inputs
and subnormal half-scale encodings; overwrite and residual forms. The archived
test was compiled in the repository and independently executed successfully.

Actual captured CPU matrices/activations also agree bit-for-bit in both store
forms, with guards. Inputs come from the retained real TinyLlama/derived
Stories15M capture, not replacement random data. The baseline arithmetic is
the Q4 part of the requested optimized profile; RMS's later contraction control
was already verified to preserve the same CPU activations and logits.

## Performance and decision

Three alternating batches of sixteen resident dispatches per mode/input,
after warmup. Include an output-buffer ordering barrier for both modes.
Allocation and shader compilation are excluded; preserve every batch and
initial observations. These short pilots are not lower-load-qualified complete
forward measurements or a production throughput estimate.

| Real matrix | Baseline GPU µs | Two-row GPU µs | Time change |
| --- | ---: | ---: | ---: |
| TinyLlama K=2048, M=2048 | 147.469 | 219.169 | +48.6% |
| TinyLlama K=2048, M=256 | 65.445 | 97.320 | +48.7% |
| TinyLlama K=2048, M=5632 | 117.234 | 103.003 | −12.1% |
| TinyLlama K=5632, M=2048 | 143.859 | 204.023 | +41.8% |
| Stories15M K=768, M=288 | 15.375 | 22.315 | +45.1% |

The ratio of sums of the four TinyLlama shape medians is **1.31541**: 31.5%
more GPU time, with losses on three of four shapes. This crosses the predeclared
stop rule: over 5% summed cost and at least three shape losses. Stop immediately
after the pilot. This simple sum is a screening statistic, not a weighted estimate
of whole-model latency. No hardware-counter diagnosis establishes the cause.

The large gate/up shape is faster in this pilot. It is retained in the evidence,
not selected afterward as a new dispatch policy. A shape-specific path would
need an independent confirmation and complete-history measurement. This run
does not establish such a path or support enabling two-row tiling globally.

No nine-batch follow-up, full-model candidate forwarding, certificate, codec or
cross-backend recovery campaign was launched. The exactness claim is limited
to the tested kernel outputs; no complete-model replay claim is made for this
rejected implementation. No new ordinary-backend ratio or change to the earlier
2.86× measured gap is inferred.

## Research archive

The rejected shader lives only in `metal/tests/fixtures/q4_two_row.metal`; the
GPU regression is `metal/tests/two_row_q4k.rs`. They are research-branch artifacts
and do not add a runtime flag or active engine dispatch. The optimized engine
code, main and default behavior remain unchanged.

Reproduce the exactness check from the repository, with host GPU access and
the existing offline Cargo cache:

```sh
cargo test --release --offline --manifest-path metal/Cargo.toml \
  --test two_row_q4k -- --ignored --nocapture --test-threads=1
```

The workspace artifact `outputs/vitni-metal-q4-two-row-2026-11-12` contains the
frozen plan and immutable baseline binary hash, both shader sources, standalone
probe/test sources and binaries, all per-batch timing observations, raw decision,
successful test/build logs, source patch and manifest. No paid compute, main
change, push, release or background-application mutation occurred. One physical
Mac/compiler and five captured inputs do not establish other GPU/model behavior
or performance for a selectively dispatched variant.
