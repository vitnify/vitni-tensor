# Separate Q4 register exchange and static metadata experiments

November 12, 2026. Research branch only; default behavior remains unchanged.
The baseline enables Q4 `no_contract` plus cooperative RMS, hidden Q6 and
attention. Both candidates change Q4 alone and are tested independently.

## Register exchange: stopped after the pilot

Replace Q4's shared-memory partial exchange and two barriers per chunk with
explicit SIMD shuffles. Retain 64-thread groups, eight physical lanes per row,
every canonical lane addition and the original adjacent-pair reduction tree.
This does not use `simd_sum` or the rejected rounding-summary algorithm.

CPU differential boundary tests pass 2,088 scalar comparisons, including
baseline controls, plus padded-row guards. Cases include subgroup/row tails,
8192-element chunk transitions, odd chunk counts, signed-zero activations and
subnormal scale encodings. Real captured matrix outputs also match CPU.

Three alternating batches of 16 resident dispatches per mode/shape give a
summed TinyLlama matrix GPU-time ratio of 1.05269, with losses on three of four
shapes. This exceeds the predeclared early-stop screen: over 5% summed cost and
at least three shape losses. The largest individual loss is about 24.2%.
Stop this candidate's performance campaign; no full-model speed or replay
claim is made for it. These are short pilot observations, not an established
production slowdown. All observations and candidate source remain archived;
the active engine has no register-exchange variant.

## Static metadata: retain behind an opt-in

For every immutable Q4 block, decode the half scales and packed scale/minimum
fields once at load. Store their sixteen separately rounded f32 products,
followed by the original 128 packed-nibble bytes: 192 bytes per resident block
instead of 144. Runtime still performs the original rounded scale-by-nibble
multiply, subtraction, activation multiply, canonical eight addition chains,
8192-element boundaries, chunk/lane trees and residual epilogue. The existing
shared-memory lane exchange remains. Contraction stays explicitly disabled.

Retain the original sixteen header bytes separately on the host so regime 3
can reconstruct the exact raw blocks. Regime switching replaces the resident
buffer rather than retaining both full raw and expanded GPU copies. A 2→3→2
switch repacks on return to regime 2; saved headers are released in regime 3.

### Real-matrix screen

CPU overwrite/residual outputs and guards match. In the nine-batch follow-up,
candidate/current GPU-time ratios are 0.859, 0.882, 0.790 and 1.086 for the four
TinyLlama shapes; the second network's Q4 down projection is 0.898. The wider
TinyLlama input is slower, so a complete history decides the net value.
Compilation/allocation are excluded from these resident dispatch timings.

### Complete replay and performance

One M3 Max, TinyLlama 1.1B Chat Q4_K_M, the original frozen 108-token codec
history with its original positions and two resets, context 64, regime 2.
Three fresh serial processes per Vitni mode; alternate forward/reverse order.
Keep first-token/first-process costs and every observation. Three fresh ordinary
Metal controls use the same inputs. The two start samples are below 7 and ten
seconds apart; every before/after one-minute load reading is below 7. This is
lower load, not an isolated GPU. Background applications remain untouched.

| Median time | Optimized baseline | Static metadata |
| --- | ---: | ---: |
| Forward, 108 steps | 1.656570 s | 1.490611 s |
| Setup | 0.509102 s | 0.608766 s |
| Fresh worker, setup/protocol/verification included | 3.007926 s | 2.940369 s |

Forward time falls 10.0182%, with three paired wins. Median fresh-worker time
falls 2.246%. The candidate's first worker is slower: 3.295506 s versus
3.108083 s, including 0.964715 s setup. It is retained in the reported samples;
no observation is filtered out. This is a model-resident speed improvement
with a setup/storage tradeoff, not uniformly faster startup.

Ordinary Metal median forward time is 0.452738 s. Same-cohort ratios are
3.6590× baseline and 3.2924× candidate. The previous attention confirmation's
3.29× used a different ordinary-control timing; multiplying it by this gain
would not establish a 2.96× result.

### Storage

TinyLlama actually selects 134 Q4 matrices:

| Allocation | Bytes |
| --- | ---: |
| Equivalent original Q4 weight buffers | 477,167,616 |
| Metadata-format GPU weight buffers | 636,223,488 |
| Saved host headers for exact restoration | 53,018,624 |

GPU Q4 buffer bytes rise 33.3%: 159,055,872 extra bytes, about 151.7 MiB.
Saved headers add about 50.6 MiB. Together this adds about 202.3 MiB over the
original Q4 allocation, or 44.4% for the Q4 portion. Apple unified memory means
the host headers consume physical memory too. Other weights, KV caches, the
caller-owned GGUF and transient construction/switch allocations are excluded;
these figures are not peak RSS or a whole-model memory percentage.

### Exactness and recovery

- All 12,416,000 candidate full-history logit values match independent CPU
  reference bytes: three TinyLlama histories plus one 64-step derived Stories15M
  history. Baseline controls match another 12,416,000 values. No tolerance or
  top-token-only substitution is used.
- Permanent GPU boundary tests pass 2,088 scalar comparisons including controls,
  plus guards. A host packing test restores exact original raw weight bytes.
- Actual 2→3→2 model switches and resets pass 384,000 direct logit-word comparisons
  per model, 768,000 across both networks; storage-layout engagement is asserted.
- CPU/Metal certificate agreement and the existing regime-3 regression pass.
- Fresh baseline/candidate ASCII encodes match CPU cover, key, coding tables and
  full-logit-row fingerprints. CPU→candidate Metal and candidate Metal→CPU
  decoding recover the original file bytes. Decoders receive only cover/key.
  These single codec observations are correctness checks, not repeated codec
  throughput estimates.

The first ordinary-control attempt used system Python and stopped on missing
NumPy after all six Vitni timing workers had completed. The remaining controls
and second-model checks used the existing benchmark virtual environment.
Completed measurements were preserved and not rerun. A test-only Rust type
error was fixed before its first execution; failed build logs remain archived.

## Use and scope

Enable before constructing `MetalForward`:

```sh
VITNI_METAL_Q4K_VARIANT=metadata
VITNI_METAL_RMS_COOPERATIVE=1
VITNI_METAL_Q6K_COOPERATIVE=1
VITNI_METAL_ATTENTION_COOPERATIVE=1
```

The metadata variant also selects cooperative Q4 and contraction-off behavior.
Keep this opt-in: one physical Mac/compiler, one performance workload and a
second small topology for numerical coverage do not validate other hardware,
larger models, long contexts, arbitrary IEEE-754 inputs or native speed parity.
No new default, arithmetic regime, CUDA path or novelty claim is introduced.

Artifact: `outputs/vitni-metal-q4-exchange-metadata-2026-11-12` in the research
workspace. It retains plans, both candidate shaders, native probes/binaries,
all pilot/timing observations, CPU-checked histories, codec files, failure logs,
source patches and manifests. The probe and boundary binary test both isolated
variants; the retained predictor implements metadata only.

From the repository, with host GPU access and the existing Cargo cache:

```sh
cargo test --release --offline --manifest-path metal/Cargo.toml \
  --test q4_metadata -- --ignored --nocapture --test-threads=1
```

The full-model switch test additionally requires `VITNI_GGUF` and the opt-ins
above. Exact worker commands, flags and immutable binary provenance are in the
artifact; runners refuse to overwrite saved measurements. Main is unchanged.
