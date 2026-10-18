# Deterministic CUDA backend — benchmark & verification

This documents `vitni-tensor-cuda`: a deterministic NVIDIA-GPU inference backend for the
engine, and a regulated-industry benchmark (credit/loan underwriting) that issues a
cryptographic **execution receipt per decision** and proves each receipt reproduces
**bit-for-bit on independent hardware**.

The headline claim Vitnify makes is *execution integrity*: the same model + input yields
the same committed trajectory, verifiable by re-running it anywhere. The question this
benchmark answers is whether that holds on a serious model (32B) on a real datacenter GPU
— and what it costs.

> All numbers below are measured, not projected. Where a baseline was **not** run (e.g.
> native llama.cpp on these boxes), it is called out rather than estimated.

## What the backend is

`vitni-tensor-cuda` mirrors the Apple-GPU backend (`vitni-tensor-metal`): it implements the
full quantized Llama/Mistral/Qwen2 forward on an NVIDIA GPU and drives it through the
engine's `inference::run_quantized_with_forward` seam. Because only the per-token forward is
swapped (the certificate/engine code is unchanged), the GPU issues **the same certificate
the CPU does** — its logits are bit-for-bit identical.

How bit-identity is held on the GPU:

- **No FMA contraction.** The `.cu` is compiled by `nvcc` at build time (`build.rs`) with
  `--fmad=false -ftz=false -arch=sm_XX`. The final SASS is baked with separate mul/add
  (matching the CPU's separately-rounded ops) and IEEE denormals, so there is **no PTX JIT
  step** that could re-introduce an `fma`. The kernels are a superset of the ones already
  proven bit-exact on a T4 in `kernels/cuda_forward_conformance.cu` (`div_sw`, `vt_expf`,
  `sqrt`, `q4k_linear`, `q6k_integer`, `rms_norm`, `attention`).
- **Fixed reduction order.** Every dot/sum uses the canonical 8-lane + fixed-tree reduction
  over 8192-element chunks (regime-2), so a reduction is bit-identical regardless of how the
  GPU schedules threads.
- **Stream ordering = Metal's barriers.** The forward kernels launch on one (default) stream,
  which serializes them, giving the same data-dependency ordering as the Metal backend's
  explicit barriers. One host↔device sync per token.
- **RoPE on the host.** The cos/sin table is computed on the CPU with the same `libm` the
  engine uses, then uploaded — so the rotation is identical.
- **Embedding on the host.** The per-token embedding row is dequantized on the CPU with the
  engine's own `dequantize_*` routines (any dtype), so the embedding is trivially identical.
- **mmap weights.** The GGUF is memory-mapped, so a 19 GB+ model needs not fit in host RAM.

Quant block types handled: **Q4_K, Q5_0, Q6_K, Q8_0** (weights), plus F32/Q4_0/Q4_K/Q6_K/
Q5_0/Q8_0 embeddings. RoPE: interleaved (Llama/Mistral) **and** NeoX split-half (Qwen2).
Optional QKV bias (Qwen2).

## Bit-identity validation

The certificate is bit-identical to the CPU engine across models **and** GPU architectures:

| Model | Arch features | GPU (CC) | Result |
|---|---|---|---|
| Mistral-7B v0.1 Q4_K_M | Llama, interleaved RoPE, no bias, head 128 | Tesla T4 (sm_75) | cert digest **bit-identical** |
| Mistral-7B v0.1 Q4_K_M | ″ | A10G (sm_86) | cert digest **bit-identical** |
| Qwen2.5-0.5B-Instruct Q4_K_M | Qwen2, **NeoX RoPE + QKV bias**, Q5_0/Q8_0 | Tesla T4 (sm_75) | cert digest **bit-identical**; 6/6 positions |
| Qwen2.5-0.5B-Instruct Q4_K_M | ″ | A10G (sm_86) | cert digest **bit-identical**; 6/6 positions |
| **Qwen2.5-32B-Instruct Q4_K_M** | Qwen2, NeoX + bias, 64 layers, hidden 27648 | A10G (sm_86) | **152064/152064 logits identical** vs independent CPU |

A single bit-identical forward pass, combined with the engine's determinism, implies an
identical full receipt; full-cert bit-identity is shown directly on the 7B and 0.5B models
(a full-generation CPU recompute of a 32B takes hours — see the tax below).

## The regulated use case: credit underwriting

`cuda/src/bin/credit_bench.rs` streams synthetic loan applications (deterministic seed →
reproducible stream) through the model as an underwriter ("Decide APPROVE or DENY and give
one short reason"), loads the model **once** and resets the KV cache per decision (the
server-realistic path), and emits a Vitnify receipt per decision. It then reproduces one
decision on an independent CPU path and compares bit-for-bit.

### Results — Mistral-7B, Tesla T4 (sm_75)

g4dn.xlarge, 8 decisions, 32 new tokens each:

| Metric | Value |
|---|---|
| Throughput | **96 decisions/hour** (0.03/s) |
| Latency p50 / p95 | 37.7 s / 38.2 s |
| Engine throughput | 2.9 tok/s |
| Receipts reproduced | **3/3 bit-identical** on an independent CPU path (full generation) |
| Deterministic CPU | 233.9 s/decision → GPU **6.2× faster** than deterministic CPU |

### Results — Qwen2.5-32B-Instruct, NVIDIA A10G (sm_86)

g5.xlarge (24 GB), 64 layers / dim 5120 / 152k vocab, 10 decisions, 32 new tokens each,
KV capped at 512 (prompts are < 100 tokens); model resident in ~21 GB:

| Metric | Value |
|---|---|
| Throughput | **47 decisions/hour** (0.013/s) |
| Latency p50 / p95 | **77.0 s / 77.8 s** |
| Engine throughput | 1.34 tok/s |
| Receipt verified | **YES** — one forward reproduced **152064/152064 logits identical** on CPU |
| Deterministic CPU | 142.1 s/forward vs GPU 0.70 s/forward → GPU **203× faster per forward** |
| Model load (mmap → GPU) | 138 s (one-time) |

Each of the 10 decisions produced a distinct, reproducible receipt (e.g. `ed720d76902b55d6…`,
`778c3ec8cca2f5e4…`). Re-running any decision — on this GPU, another NVIDIA GPU, an Apple
GPU, or CPU — yields the same digest.

## Reading the determinism tax

There are two distinct "taxes"; keep them separate:

1. **Deterministic GPU vs deterministic CPU** — measured here. 6.2× on 7B (T4), and on the
   32B a single forward is **142 s on CPU vs 0.70 s on GPU (203×)**. A full 32B decision on
   CPU is ~4 hours. Takeaway: deterministic inference at 32B is only practical on a GPU —
   which is exactly why this backend exists.
2. **Deterministic GPU vs native (non-Vitnify) GPU** — *not* measured on these boxes (no
   llama.cpp baseline was run here). The Apple-GPU tax matrix in `gpu-recompute-plan.md`
   puts deterministic Metal at ~8.6× off native Metal on TinyLlama; the NVIDIA native
   baseline is future work.

The 32B numbers above use the **correctness-first** kernels: one thread per output row, a
fixed reduction order (that fixed order is what makes them bit-identical). The 1.34 tok/s
reflects the kernel design, not an inherent cost of determinism — at 1.34 tok/s on a 32B
Q4_K_M (~20 GB/token) the engine moves ~27 GB/s, **~4–5% of the A10G's ~600 GB/s** bandwidth,
so the GPU sat ~95% idle. See **Optimizing throughput** below for the measured path from here.

## A real bug the 32B run caught

The backend was bit-identical on 7B and 0.5B across both sm_75 and sm_86, but the 32B first
produced **zero logits**. Root cause: `q6k_row` held per-super-block partials in a fixed
`float parts[64]`, while the number of super-blocks is `K/256`. The 32B's FFN-down weight
(`hidden = 27648 → 108 super-blocks`) overflowed the stack (manifesting as `inf`/`nan` right
after layer 0, localized by an L1-norm bisection). Models with `hidden < ~16 K` (Mistral-7B:
56; Qwen2.5-0.5B: 19) never hit it. Fixed by sizing `parts[256]` (covers hidden up to 65536,
i.e. 70B/405B-class). This latent bug would have shipped silently without a large-model test.

## Optimizing throughput (measured)

The ~95%-idle GPU is a kernel-design problem, not a determinism problem. Two levers, measured:

### Tier 1 — faster at the *same* digest (ship-safe, no re-anchor)

Keep the exact float canonical reduction (so every receipt is unchanged) but stop wasting the
GPU. Implemented and validated bit-identical:

- **Q6_K quantize-once**: the activation is identical across a linear's output rows, so quantize
  it once per matmul instead of per row (the lm_head previously re-quantized `x` ~150k times).
- **Q4_K warp-cooperative**: 8 threads per output row (one per reduction lane), each summing its
  lane in order, combined through the *same* fixed tree via shared memory — coalesced reads,
  8× row parallelism, identical arithmetic.

Measured on Mistral-7B (T4), **bit-identical to CPU — the receipts are byte-for-byte unchanged**:

| | before | Tier-1 |
|---|---|---|
| engine throughput | 2.9 tok/s | **5.46 tok/s** (~1.9×) |
| latency / decision | 37.7 s | **20.0 s** |

The ceiling here is structural: the canonical reduction is **8 lanes** by spec, so a single row
parallelizes at most 8-way (× the number of 8192-element chunks). Going materially faster at the
float spec would mean widening the reduction, which changes the digest.

### Tier 2 — the strategic path: integer accumulation + int8 tensor cores (new regime)

Floating-point addition isn't associative, which is the *entire* reason the order is fixed and
parallelism is capped. **Integer addition is associative and exact.** If the reduction
accumulates in integer/fixed-point (as the Q6_K path already partly does), order stops mattering:
arbitrary parallelism, cross-vendor bit-identity for free, and the **int8 tensor cores** become
usable. This is a *different numeric regime* (regime-3, integer activation) and would require a
one-time re-anchor of published digests — but it flips determinism from a tax into something free.

Measured proof (A10G, 32B's own Q4_K `w1`; CPU reference = `dot_q4k_q8_block` + `quantize_block_q8`):

- GPU integer dot == CPU integer dot: **27,648 / 27,648 bit-identical**.
- Four different GPU reductions — plain loop, `__dp4a`, warp-per-row, warp+`__dp4a` — **all produce
  the identical result** (27,648/27,648). Integer sums are associative, so the reduction method
  is free. Float can never promise this.

Int8 tensor-core proof (T4, `kernels/imma_gemm_test.cu`, a batched int8 GEMM):

| int8 GEMM M=4096 N=512 K=4096 | bit-identity vs serial integer | speed |
|---|---|---|
| **IMMA (tensor cores)** | **0 diffs / 2,097,152** | **7.8 int8-TOPS** |
| `__dp4a` | 0 diffs | 0.2 TOPS |
| serial integer | (reference) | 0.1 TOPS |

Int8 tensor cores are **bit-exact** (0 differences) and **~125× faster** than the serial integer
dot — on the same identical result. (The WMMA kernel here is naive, ~6% of the T4's ~130 int8-TOPS
peak; a tiled kernel goes several× further.) So "deterministic **and** fast" is real via integer +
tensor cores — a property float tensor cores cannot provide.

### Q4_K-IMMA kernel: correct (bit-identical), not yet perf-optimized

`kernels/q4k_imma_test.cu` is a full Q4_K linear on int8 tensor cores: on-the-fly 4-bit→int8
unpack into shared tiles, WMMA (16×16×16) for the sub-block dots, a per-sub-block integer-scale
epilogue matching `dot_q4k_q8_block`, and the fixed-tree combine across super-blocks. Batched over
N tokens (prefill). Validated **bit-identical to the integer reference** on a T4: `0/1,048,576`
diffs (M=4096, N=256). The determinism flags `--fmad=false -ftz=false -ffp-contract=off` are
required (the integer dot is exact regardless, but the float epilogue must not contract).

Status: **correct, not fast.** It measures ~0.6 effective int8-TOPS vs the 7.8 a plain int8 IMMA
GEMM reaches, because the per-sub-block scale epilogue (160 sub-blocks/tile, each a WMMA
store→shared→scale→sync) dwarfs the tiny per-sub-block tensor-core work. Making it fast is a real
MMQ-style optimization pass (register-tiled integer accumulation, larger-N weight reuse, fewer
syncs, double-buffering) that is **not yet done** — the kernel here proves correctness/bit-identity,
not production throughput.

Still outstanding for enterprise readiness: that perf pass; wiring regime-3 into the forward with
prefill batching; the one-time re-anchor of published digests; robustness across shapes/quant mixes;
and bit-identity CI across GPU archs.

### Perf pass: kernel progression (all bit-identical on a T4)

| kernel | int8-TOPS (M=4096 N=256) | vs v1 | note |
|---|---|---|---|
| v1 — one warp per 16×16 WMMA tile | 0.6 | 1.0× | good occupancy, re-reads weights once per token-tile |
| v2 — one warp per 16 rows (weight-reuse) | 0.3 | 0.4× | weights read once, but 256 warps → occupancy collapses |
| v3 — multi-warp block, shared weight staging | 0.9 | ~1.4× | best early WMMA; **epilogue-bound** (VWARPS=4 > 8 > 16 → not weight-bound) |
| v4 — register-tiled `__dp4a` (running-sum) | 2.1–2.4 | 3.3–4.3× | no `store_matrix_sync`; scales + combine in registers — but **not tensor cores** |
| **v6 — WMMA, in-register sub-block scale** | **1.7** | **~2.0×** | **tensor cores; epilogue solved** (no `store_matrix_sync` in the hot path); one-warp/block → staging-bound |
| **v7 — v6 + cooperative multi-warp (weight reuse)** | **2.0–2.1** | **2.2–2.6×** | **fastest bit-identical kernel; beats v4 dp4a 1.1–1.35×** |

Lessons, each measured:
- Weight-reuse / occupancy / tensor-core-feeding **trade off** — v2 proved reuse alone loses to occupancy; v3
  proved early WMMA stays **epilogue-bound** (~1 TOPS) because the per-sub-block `store_matrix_sync`→shared→scale
  between tiny mma pairs dominates.
- **`__dp4a` register-tiling (v4)** sidesteps the epilogue by never using tensor cores: the per-sub-block integer
  scale and the cross-super-block combine both live in registers, so there is no `store_matrix_sync` and no parts
  storage. To make the combine register-cheap, **regime-3 defines the cross-super-block combine as a fixed-order
  running sum** (still a fixed order → deterministic/bit-identical; the reference is `ref_dot_rs`). Validated
  bit-identical up to M=8192 N=2048 (0 / 16,777,216 diffs), reaching ~2.4 TOPS and scaling with batch. But v4 runs
  on the CUDA cores, so it leaves the tensor-core path on the table.

### Solving the epilogue-bound problem (v6/v7) — the tensor-core payoff

The blocker that pinned WMMA at ~1 TOPS was the per-sub-block scale: after each tiny int8 mma (a 16×16 int32
tile, one of 8 sub-blocks × `nsuper`), the integer scale `sc_j[row]` has to be applied to the accumulator. v1–v3
did it by `store_matrix_sync`→shared→scale→sync **per sub-block** (160×/tile), which dwarfs the mma. Applying the
scale **in the accumulator fragment** avoids the store entirely — but the fragment's element→(row,col) layout is
implementation-defined, so hardcoding it would re-introduce an arch dependence and threaten cross-GPU bit-identity.

**v6 solves this by deriving the layout from the hardware itself, at kernel start, using only the documented store
path.** Each lane tags its 8 accumulator elements with a unique id, does one `store_matrix_sync` to a 16×16
scratch, and reads back which `(row,col)` each id landed on — giving a `rowmap[e]`/`colmap[e]` that is correct on
*any* architecture (nothing is hardcoded; the probe measures whatever the compiler/arch actually does). Because the
fragment layout is stable per compile, every subsequent mma writes element `e` to that same `(row,col)`, so the
per-sub-block scale is applied **in registers** (`acc[e] += sc_j[rowmap[e]] * fc.x[e]`) with **no
`store_matrix_sync` in the hot path, no shared round-trip, no per-sub-block sync**. Integer accumulation is exact,
so v6 is **bit-identical to `ref_dot_rs`** (0 diffs, M=4096 N=256: 1,048,576 outputs) — on tensor cores.

v6 at one-warp-per-block is staging-bound (re-unpacks the 16 weight rows once per token-tile). **v7** puts the same
in-register-scaled WMMA inside a cooperative block: all `WN·32` threads stage the 16 weight rows **once** per
super-block and reuse them across `WN` token tiles (the MMQ weight-reuse win), each warp running its own mma +
in-register scaling. v7 is likewise **bit-identical** and is the **fastest bit-identical kernel — 2.0–2.1 TOPS,
beating the v4 dp4a kernel by 1.1–1.35×** (WN=4 optimal; WN≥8 exceeds 48 KB static shared on the T4).

The epilogue problem is **solved** (in-register scaling, bit-identical, portable). The remaining gap to the
plain-IMMA ceiling (7.8 TOPS) is no longer the epilogue — it is per-super-block **staging/unpack bandwidth** and
occupancy (v7 plateaus at ~2.1 across N=256→1024, i.e. no longer weight-reuse-limited). Closing that is the next
tier: double-buffered / `cp.async` staging, `>48 KB` shared for wider blocks, and an Ada/Hopper IMMA variant.

### The combine decision (regime-3 spec, GPU-critical)

Integrating against the ENGINE's regime-3 surfaced a key correction: `linear_q4_k_integer` combines
super-blocks with **`fixed_tree`** (inherited from regime-2's style), not the running sum v4 used. Matching it
exactly (v5 = v4's register-tiled pass-1 writing per-super-block parts to global + a `fixed_tree` combine pass)
is **bit-identical** but **slow — 0.4 int8-TOPS** (M=4096 N=512), because `fixed_tree` needs all `nsuper` parts
per output, which a register-tiled kernel can only hold in global (≈334 MB of parts traffic per GEMM → bandwidth-bound).

| kernel | combine | int8-TOPS | matches engine regime-3? |
|---|---|---|---|
| v5 — dp4a two-pass | `fixed_tree` (engine's current) | 0.4 | ✅ yes — but GPU-hostile |
| v4 — dp4a register | running sum | 2.2 | ❌ different digest |

So the **combine choice costs ~5× on GPU**. Running sum is *also* a fixed, deterministic order — it is just
register-cheap (one accumulator) where `fixed_tree` is not. Since **regime-3 is unpublished** (shipped `REGIME`
is `vitni-regime-2`; regime-3 exists only as reserved code + `NUMERIC_REGIME_CHECK`), its combine is still a
free design choice. **Recommendation: define regime-3's cross-super-block combine as a fixed-order running sum.**
That makes the fast register-tiled kernel (v4) the on-spec one. It is a deliberate regime-definition change
(touches `linear_q4_k_integer` + the pinned `NUMERIC_REGIME_CHECK`) coupled to the eventual re-anchor, so it
needs sign-off — not a silent edit.

With running-sum adopted, v4 is the production architecture (~4× v1, scaling with batch), ~7% of the T4 `__dp4a`
ceiling with clear tuning headroom (double-buffering, wider tiles, an Ada/Hopper int8-IMMA variant).

## Program to "enterprise-ready" regime-3 (sequenced)

1. **Perf pass** (engineering, multi-day): **epilogue-bound problem solved** — v7 (cooperative multi-warp WMMA with
   in-register, store-derived per-sub-block scaling) is the fastest bit-identical kernel (~2.1 TOPS, beats dp4a).
   Remaining headroom is staging/occupancy, not the epilogue: double-buffered / `cp.async` staging, `>48 KB` shared
   for wider blocks, and an Ada/Hopper IMMA variant, to approach the int8-TOPS ceiling. Gate: `q4k_imma_test`
   bit-identity at every step (held: 0 diffs through v7).
2. **Forward integration**: port v7 into the backend's batched prefill (replacing the `__dp4a` `k_q4k_rs_gemm`) in
   `run_quantized_with_prefill`, keeping decode (N=1) on the existing path or a GEMV; CPU reference stays
   `linear_q4_k_integer_rs`. Expected modest end-to-end gain until the staging tier lands (prefill is also bound by
   unbatched Q6_K/attention/rms).
3. **Re-anchor** — a **release/governance decision, not a silent code change.** Regime-3 produces different
   digests, so every published anchor/receipt must be regenerated in a coordinated, versioned migration;
   external verifiers of old receipts must be told. Requires explicit product sign-off before execution.
4. **Robustness + CI**: all shapes and quant mixes (Q4_K/Q5_0/Q6_K/Q8_0, tied embeddings, GQA, NeoX),
   and a bit-identity gate on Apple + NVIDIA (sm_75/86/89/90) runners on every kernel change.

## Caveats (stated plainly)

- **The GPU is an A10G (24 GB), not an L40S/H100 "flagship."** The 48 GB L40S (g6e) was
  capacity-blocked in every reachable AZ, and the account's GPU quota is 4 vCPU in a single
  region. The A10G is a legitimate datacenter inference GPU, and the determinism/receipt
  results are GPU-architecture-agnostic (proven on sm_75 and sm_86), but the throughput
  numbers are an A10G's, not a flagship's.
- **Throughput is correctness-first, not optimized** (see above).
- **The native (non-Vitnify) GPU baseline was not run here**, so no deterministic-vs-native
  tax is claimed for NVIDIA.

## Reproduce

On a CUDA box (Deep Learning Base AMI; set the arch for the card):

```bash
# arch: T4=sm_75, A10G=sm_86, L4/L40S=sm_89, H100=sm_90
export CUDA_HOME=/usr/local/cuda VITNI_CUDA_ARCH=sm_86

cd vitni-tensor/cuda
cargo build --release --features parallel --bin credit_bench

# bit-identity (end-to-end certificate) on a 7B-class model:
VITNI_GGUF_7B=/path/mistral-7b-Q4_K_M.gguf \
  cargo test --release -- --ignored --nocapture cuda_certificate_matches_cpu_7b

# the credit benchmark (cap KV on a 24 GB card; N_VERIFY=1 does the bit-identity check):
VITNI_GGUF=/path/Qwen2.5-32B-Instruct-Q4_K_M.gguf VITNI_MAX_SEQ=512 \
  N_DECISIONS=10 N_VERIFY=1 ./target/release/credit_bench
```

The benchmark exits non-zero if the receipt does not reproduce bit-identically on CPU.
