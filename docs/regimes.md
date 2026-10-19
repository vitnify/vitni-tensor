# Numeric regimes

A **regime** is the exact arithmetic contract the engine computes under. It is bound into
every receipt — the `REGIME` string is length-prefixed into the `vitnify-receipt v2`
digest (`src/cert/builder.rs`) — so a receipt is only meaningful, and only reproducible,
*relative to its regime*. Two parties who reproduce the same model + input under the same
regime get a byte-identical digest on any hardware.

vitni-tensor defines two live regimes. **Both are fully deterministic and reproducible
bit-for-bit across devices.** They are two different numeric *contracts*, not a quality
hierarchy — "which deterministic answer," not "deterministic vs not."

## regime-2 — `vitni-regime-2` (shipped default, anchored)

- **Arithmetic.** Q4_K linears via the float-activation fused kernel; fixed-order fp32
  reductions for rms / softmax / attention / matmul (the canonical reduction); exact-integer
  Q6_K. Weights are dequantized and multiplied by **float** activations.
- **Determinism.** Bit-identical across CPUs (Apple, Graviton, Intel, AMD — two ISAs, two
  compiler builds) **and** GPUs (Apple Metal, NVIDIA CUDA sm_75/sm_86): the GPU backends emit
  the same certificate as the CPU, byte for byte.
- **Status.** This is the shipped default and the regime the published anchors in
  [`regime-manifest.json`](../regime-manifest.json) are computed under. Verifiable against
  those anchors today via [`tools/check-anchors.sh`](../tools/check-anchors.sh).
- **Use when** you want the canonical, anchored result and maximum float-activation precision —
  the default for CPU, and for GPU correctness.

## regime-3 — integer-activation + running-sum (reserved, opt-in)

- **Arithmetic.** The k-quant linears (Q4_K, Q6_K) run in **exact integers**: activations are
  quantized to int8, the dot is an integer dot, and the cross-super-block combine is a
  fixed-order **running sum**. Every other op is identical to regime-2.
- **Why it exists.** Integer accumulation is associative, so the result is **independent of the
  order** the hardware sums it. That is exactly what makes it bit-identical on **integer tensor
  cores** (WMMA / `mma.sync`), where float cannot be — float addition isn't associative, so
  float tensor cores diverge across hardware. Regime-3 is the deterministic high-throughput GPU
  path (see [cuda-benchmark.md](cuda-benchmark.md)).
- **Determinism.** Bit-identical across CPU, NVIDIA **CUDA** (T4, A10G), and Apple **Metal**
  (M3 Max) — cross-vendor, same breadth as regime-2 — and deterministic by construction on any
  device implementing the same integer ops. Determinism is **not** reduced relative to regime-2 —
  if anything it is *more* hardware-robust, because it does not depend on reduction order.
- **Fidelity.** The int8-activation numerics are the same ones standard GPU quantized inference
  uses (e.g. llama.cpp's MMQ path). The arithmetic is slightly lower precision than regime-2's
  float activations, but it is a *different deterministic answer*, not a noisy one — and it is
  faithful to how the quantized model is normally served. As an **attestation**, regime-3 is
  exactly as high-fidelity as regime-2: it verifiably records precisely what ran.
- **Status: reserved / not yet anchored.** The bound `REGIME` string is still `vitni-regime-2`,
  and there are no published regime-3 anchors. A regime-3 result is reproducible by re-running
  the engine in regime-3, but it does **not** verify against the current (regime-2) anchors.
  Making regime-3 a *published* standard is a one-time, versioned **re-anchor**: bump `REGIME`
  to `vitni-regime-3`, regenerate `regime-manifest.json`, and tell verifiers of regime-2
  receipts that the era changed (support both). That is a deliberate release decision, not a
  silent code change. Entry points: `inference::run_quantized_regime3` and the CUDA backend's
  `CudaForward::new_regime(…, 3)`.
- **Use when** you need maximum GPU tensor-core throughput at scale and accept the
  int8-activation precision — as most production quantized inference does.

## Choosing

| | regime-2 | regime-3 |
|---|---|---|
| activations | float | int8 |
| determinism | bit-identical (CPU / Metal / CUDA) | bit-identical (CPU / Metal / CUDA) |
| verifiable vs *published* anchors | yes, now | after the re-anchor |
| GPU integer tensor cores | no (float path) | yes |
| best for | canonical / anchored, max precision, CPU | GPU-at-scale throughput |

Both are honest, deterministic, and externally reproducible. A receipt names its regime; a
verifier reproduces that regime and checks against that regime's anchors. Keeping both lets the
same engine serve a max-fidelity/anchored path and a GPU-native throughput path without either
one compromising the integrity claim.

> History: `regime-1` was the pre-2026-08 reduction order (superseded by regime-2's canonical
> reductions). Only regime-2 and regime-3 are live.
