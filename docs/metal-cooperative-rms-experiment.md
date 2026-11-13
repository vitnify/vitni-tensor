# Opt-in cooperative Metal RMS normalization

This experiment parallelizes the existing RMS operation without changing its
numerical profile. Set `VITNI_METAL_RMS_COOPERATIVE=1` before constructing
`MetalForward`. `rms_cooperative_enabled()` reports whether the candidate is
selected. It is disabled by default and selected only for widths 1 through 8192,
the existing Metal RMS path's single-canonical-chunk range. Larger widths keep
the existing path; this experiment adds no wider-row conformance claim.

The Q4_K opt-in flag is independent. To use both research optimizations, set
`VITNI_METAL_Q4K_COOPERATIVE=1` as well. RMS arithmetic is shared by regimes 2
and 3; both regimes are checked with the RMS flag enabled.

## Mechanism

One 64-thread group owns one row. Eight threads compute the original eight
canonical sum-of-squares chains, each visiting exactly its original element
sequence. After a threadgroup barrier, thread 0 combines those partials through
the original fixed tree and computes the same software divisions and square
root. A second barrier shares the scale. All 64 threads write independent
output elements, retaining `(x * scale) * weight` as two rounded multiplies.

The model stays resident and the production graph retains one compute encoder
and one command buffer per token. Existing resource barriers remain. No
profiling instrumentation is enabled for the application timing comparison.

## Validation and workload

`metal/tests/cooperative_rms.rs` compares the actual CPU Tensor RMS operation
with both Metal kernels over 18 widths from 1 to 8192, three seeds, three
positive epsilons, multiple rows, signed zeros and varied finite magnitudes.
It checks out-of-place and in-place stores, padded row groups and output guards.
There are 1,974,528 scalar output-bit-pattern comparisons plus guard checks.

The local artifact `vitni-metal-rms-2026-11-11` freezes three existing Subtext
steganographic codec fixtures: an empty frame, 13-byte ASCII and 14-byte
binary/Unicode. Fresh controls use the stock engine, cooperative Q4_K only,
cooperative Q4_K plus RMS, and ordinary llama.cpp Metal. Each encoding cell has
three serial fresh processes with mode order reversed on alternating rounds.

All Vitni coding tables, full-logit-row fingerprints, token/reset histories,
covers and keys are compared with frozen CPU encodes. Those CPU encodes are
retained references, not newly measured encodes. Fresh CPU-source-to-candidate
Metal and candidate-Metal-to-CPU decodes check actual file bytes. Ordinary
same-backend decoder controls are also checked. A separate uninstrumented
matched-history comparison replays the same 108 inputs and resets through all
four modes, avoiding token-count differences in the ordinary comparison.

The test commands require host Metal GPU access:

```sh
cargo test --release --manifest-path metal/Cargo.toml --test cooperative_rms \
  -- --ignored --nocapture --test-threads=1
VITNI_METAL_Q4K_COOPERATIVE=1 VITNI_METAL_RMS_COOPERATIVE=1 \
  VITNI_GGUF=/path/to/tinyllama.gguf \
  cargo test --release --manifest-path metal/Cargo.toml --lib \
  tests::metal_certificate_matches_cpu -- --exact --ignored --nocapture
```

## Measured results: November 11, 2026

On an Apple M3 Max (36 GiB, macOS 27.2.1), TinyLlama 1.1B Chat Q4_K_M:

| Fixture | Stock work (s) | Q4 only (s) | Q4 + RMS (s) | Ordinary Metal (s) |
| --- | ---: | ---: | ---: | ---: |
| Empty frame | 9.127 | 6.054 | 4.961 | 1.058 |
| ASCII, 13 bytes | 6.181 | 4.152 | 3.369 | 0.591 |
| Binary/Unicode, 14 bytes | 3.155 | 2.131 | 1.746 | 0.602 |

These are medians of three fresh encodes per cell. Work includes model
forwarding, protocol traffic, filtering, probability-table construction and
tracing; it excludes imports, setup and teardown. Ordinary can generate a
different number of tokens for the same file, so this table compares complete
encoding workflows. Cover sizes are unchanged between Vitni modes; these tiny
steganographic fixtures do not demonstrate general-purpose compression.

For ASCII, RMS removes another 18.87% of encoding work relative to Q4 only;
the combined reduction from stock is 45.49%. Full fresh-process wall time is
10.435 s stock, 8.653 s Q4 only and 7.869 s combined. Native forwarding within
the codec is 5.425 s, 3.401 s and 2.610 s respectively.

The separate matched-history replay has medians of 6.057 s stock, 4.284 s Q4
only, 3.803 s combined and 1.068 s ordinary. This is an 11.23% incremental
reduction in forwarding time. It ran later under substantial background load;
the ordinary control is also slower than in the earlier profiling experiment.
Do not infer improved native parity by comparing ratios across those sessions.

All 45 fresh codec workers passed (36 encodes and nine decodes), and every
decoded file recovered its original bytes. All 3553 fresh Vitni codec logit-row
fingerprints and corresponding tables/histories match their CPU references.
The 12 matched-history workers passed another 972 Vitni row comparisons. The
RMS boundary/store test, full-model CPU/Metal certificate test and existing
regime-3 regression passed; the profiling-feature build also checks successfully.

All observations are retained, including a slower third ASCII candidate run
(3.821 s work versus 3.369 s and 3.338 s) and process-startup outliers. A late
snapshot records load average 28.47 on 14 physical CPU cores. Causes of
individual outliers were not established. These timing estimates are preliminary,
and the matched-history results show that the size of the gain varies with the
test conditions. Native-speed parity remains unproven. One host, one small model
and short fixtures do not establish larger-model, CUDA, long-context,
concurrency or production-load performance. The default remains unchanged.
