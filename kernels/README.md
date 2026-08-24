# vitni-gpu-ref — deterministic GPU matmul, proven against the CPU contract

This is the GPU half of the numerical contract. It closes the gap that used to
read *"the reproducible path is CPU-only; a GPU kernel is a contract stub, not
shipped code."*

The `matmul` reduction in `vitni-tensor` (`ops::quant::canonical_dot` →
`canonical_chunk` → `fixed_tree`) **is** the certificate's definition of "the
same computation". A GPU can run it — for real speed — and still land on the
exact same bits, **if and only if** the kernel obeys two rules:

1. **Order is fixed by the contract, never by the hardware.** Element → lane is
   `i % 8`; lanes and chunks reduce through a fixed pairwise tree in index
   order. No atomics, no warp shuffles, no hardware-chosen accumulation order.
2. **No FMA contraction.** The multiply and the add stay separately rounded.
   Fusing `x*w + acc` into one `fma` rounds once instead of twice and moves the
   bits.

## What is proven here (and where)

`kernels/canonical_matmul.metal` is a bit-exact port of the CPU reduction.
`src/main.rs` is the conformance harness. On this machine (Apple M3 Max, Metal):

```
== ground truth ==
  CPU replica FNV over (4,64,4) pin vector: 0x8a428433686d13af
  PINNED_MATMUL_HASH (from matmul.rs):     0x8a428433686d13af   # replica IS the contract

== Metal, fastMathEnabled = false ==
  GPU FNV over (4,64,4) pin vector:        0x8a428433686d13af   # == pin
  GPU vs CPU bit-for-bit on pin vector:    IDENTICAL

== bit-for-bit sweep (Metal fast-math OFF vs CPU) ==   # all shapes, max ULP 0
  [2x3x2] .. [4x14336x4] (Mistral FFN K, 2 chunks) .. [7x333x5] tail path   OK

== control: Metal fast-math ON (fma allowed), [4x14336x4] ==
  exact-bit 1/16, max ULP 64   # DIVERGES — the gap fast-math-off closes
```

Run it: `cargo run --offline` (exits non-zero on any divergence, so it works as
a CI gate).

The **ground-truth anchor** is what makes this airtight: the CPU code in
`src/main.rs` reproduces the shipped pin `0x8a428433686d13af` from
`matmul.rs::matmul_reduction_bits_are_pinned`, so the replica is the contract
byte-for-byte; the GPU is then measured against it.

The **control** (same kernel, fast-math on) is the honest negative: a
non-conformant GPU kernel — which is what stock cuBLAS / PyTorch-CUDA /
llama.cpp-CUDA are — does *not* match. Determinism is a property of *this*
kernel with *these* flags, not of "running on a GPU".

## Status per hardware target

| Target | Kernel | Verified bit-exact vs CPU contract |
|---|---|---|
| Apple GPU (Metal) | `canonical_matmul.metal` | **Yes** — M3 Max, this harness |
| NVIDIA (CUDA) | `canonical_matmul.cu` | **Yes** — Tesla T4 (CC 7.5), CUDA 13.2 (see `RESULTS-nvidia-t4.txt`) |

All three — CPU, Apple M3 Max (Metal), NVIDIA T4 (CUDA) — land on the same pin
`0x8a428433686d13af`. `kernels/cuda_conformance.cu` is the self-contained NVIDIA
harness (host reference + both kernels); build it `--fmad=false -ftz=false` and
never `--use_fast_math`. The shipped kernel additionally uses `__fmul_rn` /
`__fadd_rn`, so it stayed bit-exact even under `--fmad=true` — the control
(bare `*`/`+` with fusion on) diverged by up to 448 ULP, which is exactly the
gap the contract closes.

## Forward-pass ops (cross-vendor) — `conformance-forward/`

Extending bit-identity from matmul to a full forward means every op must agree
across **CPU + Apple GPU + NVIDIA GPU**, not just one vendor. Proven so far
(`kernels/conformance-forward`, run on Apple M3 Max):

- **Software f32 division** (`div_sw`): integer reciprocal seed + FMA Newton
  refine + FMA correction — only ops that are correctly-rounded on every device.
  **Bit-identical CPU↔Metal: 8,000,000/8,000,000**, and correctly-rounded
  (0 ULP vs IEEE). Metal's *hardware* divide is NOT correctly-rounded (~4.3% of
  `a/b` differ), so a cross-vendor kernel must never use it.
- **`expf`** (softmax + SiLU): a bit-exact port of `libm` 0.2.16 `expf`
  (f32-only; no f64, so it runs on Metal) on top of `div_sw`. Bit-identical to
  `libm::expf` for every normal-range output; the only divergence is the
  denormal tail (exp(x), x < ~-87), which Apple GPUs flush to zero — harmless
  (far below the ULP of any downstream sum) and exact on NVIDIA with `-ftz=false`.

**The load-bearing lesson: "fast-math off" is not portable.** Metal still
contracts `a ± b*c` into an FMA where the CPU (Rust never auto-FMAs) keeps them
separate — a scattered 1-ULP divergence. Bit-identity therefore requires
*explicitly* controlling contraction at every fused site. `volatile` on each
product forces f32 rounding before the add/sub, matching the CPU and preserving
the existing digests (no regime change). The alternative — spelling every site
with explicit `fma` on all three platforms — is faster but changes the numbers
(a regime change / new anchors).

## Scope / roadmap

Done: F32 `matmul`, software division, `expf` — all cross-vendor. Next kernels
toward an end-to-end deterministic GPU forward: the quantized hot path
(`canonical_dot_q4k_fused` — same pinned reduction over a dequantized chunk,
division-free), then `rms_norm` / `softmax` / `silu` (reductions + the proven
`div_sw`/`expf`), `rope` (reads a CPU-precomputed cos/sin cache), embedding,
argmax, and attention — then a full TinyLlama forward whose certificate digest
matches across CPU + Apple GPU + NVIDIA.
