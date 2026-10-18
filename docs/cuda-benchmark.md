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

The epilogue problem is **solved** (in-register scaling, bit-identical, portable). The remaining gap on the T4 was
staging bandwidth, which the next tier closes.

### The wide tile (v8/v9) — cashing in the tensor-core payoff on a real inference GPU

v7's residual loss was that it re-stages the activations **once per M-tile** (at M=4096 that is 256× redundant).
**v8** fixes it with a wide `BM×BN` block that stages BM weight rows **and** BN activation cols once per super-block
and reuses them across all `(BM/16)·(BN/16)` tiles; each warp owns several tiles. **v9** is the same kernel in
*dynamic* shared memory, so `BM×BN` can exceed the 48 KB static cap (opt-in up to ~100 KB on sm_80+). Both keep the
v6 in-register, store-derived scaling and are **bit-identical to `ref_dot_rs`** (0 diffs through M=4096 N=1024).

Measured on an **A10G (sm_86)**, M=4096 N=256 K=4096 — a real datacenter inference GPU, not the weak T4:

| kernel | int8-TOPS | vs dp4a (v4) |
|---|---|---|
| v4 — register-tiled `__dp4a` | 7.3 | 1.0× |
| v7 — cooperative WMMA | 4.8 | 0.66× |
| **v8 — wide WMMA (64×64, static shared)** | **10.2** | **2.07×** |
| v9 — wide WMMA (128×128, dynamic shared, 87 KB) | 10.6 | 2.17× |

The wide tile is the lever: v8 ≈ v9, i.e. staging amortization is captured by 64×64 and going wider barely helps
(v9 plateaus; 128×256 at 124 KB exceeds the A10G's ~100 KB limit). The WMMA tensor-core GEMM is now **~2× the dp4a
kernel and bit-identical** — the tensor-core payoff, realized. (On the A10G, `__dp4a` is itself strong, so v7's
per-M-tile restaging actually lost to it; v8's wide staging is what wins.)

### Wired into the backend prefill (end-to-end, bit-identical)

v8 is ported into `cuda/vt_cuda.cu` as `k_q4k_rs_gemm_wmma` (token-major `out[N][M]`, `acc` residual, ragged
M/N via zero-staging) and is the Q4_K GEMM in the batched prefill (`pg()`); `VITNI_Q4K_DP4A=1` falls back to the
dp4a kernel for A/B measurement. Validated on an A10G with TinyLlama-1.1B Q4_K_M:

- **GEMM vs engine `linear_q4_k_integer_rs`: 0 diffs of 360,448** (per-token, M=5632 N=64 K=2048).
- **Full prefill cert bit-identical**: CPU regime-3 `[9d,20,db,01]` == GPU r3+prefill `[9d,20,db,01]`.
- **Prefill (64 tokens): per-token 2.27 s → batched 0.61 s = 3.72×** (dp4a: 0.67 s = 3.39×).

**Finding (honest, and it redirects the effort):** at the GEMM level the WMMA kernel is ~2× dp4a, but end-to-end it
moves the prefill only 0.67→0.61 s (~10%, 3.39×→3.72×). The batched-prefill **architecture** is the big e2e lever
(~3.4×); the Q4_K GEMM is **no longer the prefill bottleneck**. On TinyLlama the remaining time is the **unbatched
attention / RoPE / RMS loops and the non-tiled Q6_K GEMM**. So the wide WMMA GEMM is the right kernel and matters
most on **GEMM-dominated** workloads (large models, larger prompt batches); for TinyLlama-class prefill the next big
win is **batching attention/rope/rms + tiling Q6_K**, not further GEMM tuning.

Levers tried and **not** pursued, with the reason: wider-than-64×64 blocks (v9, the `>48 KB` item) — built,
measured, plateaus. Double-buffered / `cp.async` staging and an `mma.sync m16n8k32` (K=32/instruction, the
"Ada/Hopper IMMA") path would push the GEMM past 2× further, but the data above shows that is off the critical path
for this prefill; they belong to a GEMM-bound (large-model/large-batch) pass, and `mma.sync` could not be validated
on sm_89/90 here (the L40S stayed capacity-blocked). The wide-WMMA kernel is forward-compatible across sm_75→90.

### Prefill critical-path pass: batch the rest, and the WMMA win surfaces on large models

The "~10%" finding above was a **TinyLlama artifact, not a general truth.** GEMM share of a forward scales with model
size: per layer per token, linears ∝ `dim²` while attention ∝ `dim·seq`, so the ratio GEMM/attention ≈ `dim/seq`.
On a small model the GEMM is small and other work dominates; on a large model (your 32B credit workload) the linears
are the overwhelming majority, so the 2× Q4_K GEMM surfaces almost directly. Each optimization is itself
model-agnostic (it speeds *its* op for any model); which op dominates is what's model-dependent. So the plan is to do
all of them — each owns a regime. Three bit-identical structural fixes to the batched prefill (all validated on an
A10G, certs unchanged):

1. **Batch attention / RoPE / QKV-bias.** The prefill looped these `P` times (host-side launch overhead); RoPE
   already had a `seq` param, attention/bias just needed the `p` loop folded into the grid. Same per-(position,head,
   element) arithmetic → bit-identical. TinyLlama prefill 3.72× → **6.10×**.
2. **Output head for the last position only.** Prefill computed the `[vocab][dim]` head for all `P` positions but only
   position `P-1`'s logits are ever read — pure `P×` waste, worst on large-vocab models. Computing only the last
   position is correctness-preserving (unused logits). TinyLlama 6.10× → **20.74×**; Qwen2.5-7B (vocab 152064) 1.16×
   → **2.13×**.
3. **Tile the Q6_K GEMM.** The Q6_K FFN-down ran on the non-tiled one-thread-per-output kernel (re-reads weights per
   token). Q6_K sub-block `j` = positions `[16j,16j+16)`, scale `scb[j]`, no min term — structurally the Q4_K kernel
   with 16 subs of 16, so `k_q6k_rs_gemm` tiles it with dp4a weight-reuse (bit-identical to `linear_q6_k_integer_rs`).
   Qwen2.5-7B 2.13× → **20.64×** (it was ~90% of the batched time).

**A real bug this surfaced:** the batched `pg()` treated the weight dtype as a boolean `is_q6k`, so Q8_0/Q5_0 tensors
(e.g. Qwen's output head) were routed to the Q6_K kernel and silently corrupted — never caught because prefill had
only been tested on pure-Q4_K/Q6_K models (Mistral/TinyLlama). The old *unbatched* loops reproduced the same wrong
digest, confirming it pre-dated the batching. Fixed `pg()` to dispatch the real kind (Q8_0/Q5_0 loop the per-token
float kernels). Qwen2.5-0.5B prefill cert now `[80,b1,dd,16]` == CPU regime-3.

**The scaling confirmed (A10G, 64-token prefill, batched vs per-token; WMMA vs the dp4a fallback):**

| model | dim / vocab | batched prefill | WMMA vs dp4a (e2e) |
|---|---|---|---|
| TinyLlama-1.1B | 2048 / 32000 | **22.9×** | 1.16× |
| Qwen2.5-7B | 3584 / 152064 | **20.6×** | **1.67×** (0.306 s vs 0.512 s) |

The WMMA-vs-dp4a gap **grows with model size** (1.16× → 1.67×): once the other work is batched/tiled away, the Q4_K
linears are a large share on the bigger model, so the tensor-core GEMM's 2× surfaces end-to-end — the large-model
(product) path. All of this is bit-identical: every change preserves the regime-3 cert (`A/B` toggles
`VITNI_Q4K_DP4A`, `VITNI_Q6K_SIMPLE`, `VITNI_PREFILL_UNBATCHED` kept for measurement); decode (N=1) still uses the
per-token path.

### GEMM-depth levers (the scale-up path): mma.sync + cp.async

For GEMM-dominated scale-up (large models, large serving batches) the Q4_K GEMM throughput still matters, so the two
remaining kernel levers were built and validated bit-identical on an A10G (`kernels/q4k_imma_test.cu`):

- **v11 — `mma.sync.m16n8k32.s8`.** WMMA int8 is `m16n16k16` (K=16), so a 32-element Q4_K sub-block needs two wmma
  mma + bookkeeping. The PTX `mma.sync m16n8k32` does the whole sub-block in **one** instruction (half the issue
  count). The Ampere s8 fragment layout is fixed/documented, so A/B load as contiguous 4-byte reads from shared and
  the per-sub-block scale is applied to the 4 output registers in-register. Wide `MM×MN` staging as in v8.
- **v12 — v11 + `cp.async` double-buffering.** v11 is still staging-bound (scales with N). cp.async async-copies the
  next super-block's raw packed weights + int8 activations into a second shared buffer while the current one
  computes, hiding the global-read latency; the cheap nibble unpack then runs shared→shared. Dynamic shared (~69 KB),
  sm_80+.

| kernel (A10G, M=4096) | N=256 | N=1024 | vs dp4a |
|---|---|---|---|
| v8 — WMMA wide | 10.2 | 10.8 | ~2.1× |
| v11 — `mma.sync m16n8k32` | 11.3 | 13.0 | 2.3–2.6× |
| **v12 — mma.sync + cp.async 2×-buffer** | **13.9** | **17.0** | **2.8–3.4×** |

All bit-identical to `ref_dot_rs` (0 diffs). v12 at ~17 int8-TOPS is ~12% of the A10G's int8 peak — the deterministic
Q4_K GEMM is now **3.4× the dp4a kernel**. Not pursued: Ada/Hopper-specific FP8 / `wgmma` paths — those need an
sm_89/90 box (the L40S stayed capacity-blocked) and a separate (non-bit-identical) numeric regime.

### Wired into the backend (arch-conditional, live and bit-identical)

v12 is ported into `cuda/vt_cuda.cu` as `k_q4k_rs_gemm_mma` (token-major `out[N][M]`, `acc` residual, ragged-`N`
handled, `M % 64 == 0` host-enforced) and the `pg()` dispatcher now picks the Q4_K kernel by device capability:
**dp4a (A/B) < wide-WMMA (portable, sm_75+) < mma.sync+cp.async (sm_80+)**. The mma body is behind
`#if __CUDA_ARCH__ >= 800`, so an sm_75 build compiles it to an empty stub (never launched — the T4 keeps WMMA);
an sm_80+ build gets the real kernel and the host selects it on `p.major >= 8`. Compile-checked on both `sm_75`
(stub) and `sm_86` (real). A/B env toggles: `VITNI_Q4K_WMMA`, `VITNI_Q4K_DP4A`.

Validated on an A10G — certs unchanged with the mma path active:

- **TinyLlama cert `[9d,20,db,01]` == CPU regime-3**; prefill 26.9× (was 22.9× on WMMA).
- **Qwen2.5-0.5B cert `[80,b1,dd,16]` == CPU regime-3** (NeoX rope + QKV bias + GQA).
- **Qwen2.5-7B prefill (64 tok): dp4a 12.4× → WMMA 20.6× → mma+cp.async 24.9×** — the sm_80+ path is auto-selected
  and is 1.21× over WMMA / 2.02× over dp4a end-to-end.

So the scale-up kernel is not just proven in the bench — it is the deterministic default on Ampere+ hardware, with
the portable WMMA kernel still covering the T4. Remaining (Ada/Hopper only): FP8/`wgmma`, gated on an sm_89/90 box.

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

1. **Perf pass** — **DONE.** Epilogue solved (v6/v7), then the wide tile (v8/v9) made the WMMA tensor-core GEMM
   **~2× the dp4a kernel** on an A10G (10.2 int8-TOPS, bit-identical), the staging amortization captured by a 64×64
   block. The `>48 KB` wider-block lever (v9, dynamic shared) was built and measured — it plateaus. Double-buffered /
   `cp.async` and `mma.sync m16n8k32` (Ada/Hopper IMMA) remain as GEMM-bound-case levers but the integration below
   showed they are off the critical path for this prefill (and `mma.sync` could not be validated on sm_89/90 here).
   Gate held: `q4k_imma_test` bit-identity at every step (0 diffs through v9, M=4096 N=1024).
2. **Forward integration** — **DONE.** v8 is wired into the backend as `k_q4k_rs_gemm_wmma` and is the Q4_K GEMM in
   the batched prefill (`pg()`; `VITNI_Q4K_DP4A=1` falls back to dp4a). GEMM bit-identical to the engine
   `linear_q4_k_integer_rs` (0 diffs), full prefill cert bit-identical (CPU r3 == GPU r3+prefill), prefill **3.72×**
   (A10G, TinyLlama, 64 tok; dp4a 3.39×). The batching architecture is the big e2e lever (~3.4×); the Q4_K GEMM is no
   longer the prefill bottleneck, so the next e2e win is **batching attention/rope/rms + tiling the Q6_K GEMM**, not
   more GEMM tuning. Decode (N=1) still uses the per-token path.
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
