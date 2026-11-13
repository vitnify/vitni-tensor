# Opt-in cooperative Metal Q4_K experiment

This research branch tests an implementation change to **regime-2**, not a new
numerical regime. Set `VITNI_METAL_Q4K_COOPERATIVE=1` before constructing
`MetalForward` to select the experimental kernel. The default remains the
existing kernel. Regime-3 and Q6_K/Q8_0/Q5_0 are unaffected.

The candidate maps the eight existing canonical accumulation chains to eight
physical GPU threads. Each chain retains its original element order, separately
rounded products and additions, and 8192-element chunk boundaries. A designated
thread combines the eight partials through the original fixed tree, followed by
the original tree over chunks. Sixty-four-thread groups process eight rows;
inactive padded rows participate in barriers without accessing model buffers.

Required checks, before interpreting performance:

1. Compare baseline Metal and candidate against the actual CPU Q4_K function at
   row tails, chunk boundaries, odd chunk counts, large reduction widths, and
   both ordinary and residual-accumulating stores.
2. Reproduce complete TinyLlama CPU/Metal certificates with the candidate enabled.
3. Run the real Subtext encoder/decoder in fresh processes. Check source bytes,
   every coding table and every full-logit-row fingerprint across CPU, baseline
   Metal and candidate Metal, including a rolling-context reset.
4. Alternate baseline/candidate order for timing. Report work, native forward,
   setup and full process wall time separately. Compare ordinary llama.cpp under
   the same codec and model. Preserve failed results and the original corpus.

Kernel diagnostics include command encoding, submission and waiting; they are
not hardware-counter-only GPU execution measurements. A kernel microbenchmark
alone is insufficient to claim an application speedup. One model/Mac does not
establish general model or hardware coverage. No paid resources are needed.

Commands (run with host GPU access):

```sh
CARGO_HOME=/path/to/cached/cargo-home cargo test --release --offline \
  --manifest-path metal/Cargo.toml --test cooperative_q4k \
  -- --ignored --nocapture --test-threads=1
VITNI_METAL_Q4K_COOPERATIVE=1 VITNI_GGUF=/path/to/tinyllama.gguf \
  cargo test --release --offline --manifest-path metal/Cargo.toml \
  metal_certificate_matches_cpu -- --ignored --nocapture --test-threads=1
```

## Measured result: November 11, 2026

On one Apple M3 Max (36 GiB, macOS 27.2.1), TinyLlama 1.1B Chat Q4_K_M,
the existing Subtext steganographic encoder produced the following medians of
three fresh processes per mode. Both Vitni paths used the same compiled binary;
the construction-time flag selected the kernel. Ordinary llama.cpp used the same
model and codec, but can generate a different number of tokens.

| Payload | Baseline work (s) | Cooperative work (s) | Ordinary Metal work (s) |
| --- | ---: | ---: | ---: |
| Empty frame | 9.111 | 6.060 | 1.048 |
| ASCII, 13 bytes | 6.157 | 4.115 | 0.585 |
| Binary/Unicode, 14 bytes | 3.156 | 2.128 | 0.599 |

Work includes forwarding, protocol traffic, filtering, probability-table
construction and tracing; it excludes setup, Python imports and process teardown.
For the 13-byte case, full fresh-process wall time was 10.425 s baseline,
8.495 s cooperative and 4.338 s ordinary. Native forwarding fell from 5.405 s to
3.365 s. The change removed about 33% of encoding work across these fixtures;
the cooperative path still took 7.03 times ordinary encoding work on the ASCII
fixture. This improves the measured cost without eliminating the tradeoff.

All 48 fresh codec workers succeeded: 27 repeated Metal encodes, three CPU
encodes, 15 Vitni cross-backend decodes, and three ordinary same-backend decoder
controls. Every tested decode recovered the original file. CPU, baseline Metal
and cooperative Metal agreed on every captured full-logit-row fingerprint,
coding table and model history, and produced identical cover text and keys.
Both Vitni paths emitted the same cover sizes (275, 206 and 177 bytes); these
small steganographic fixtures do not demonstrate general-purpose compression.

The two explicit GPU tests passed, including 1,584 CPU-referenced scalar-output
bit comparisons plus padded-row guards. The full model CPU/Metal certificate
test and existing regime-3 regression passed with the opt-in flag set. The root
release suite passed 153 tests with 29 ignored. The diagnostic dispatch benchmark
showed 1.52–1.84 times baseline throughput at the four tested matrix shapes;
those numbers include host dispatch and waiting, and do not replace the codec
measurements above.

This is one model on one host, three encoding repetitions and one observation
per decoder control. Large synthetic matrix shapes are not validation of a
larger real model. CUDA, a second host, long payloads, production concurrency and
model-quality effects were not tested here. The default remains unchanged.
The complete local experiment artifact is named
`vitni-metal-optimization-2026-11-11`; it contains the frozen plan, scripts,
all timing observations, failed setup/report attempts, traces and command logs.
