# Changelog

All notable changes to `vitni-tensor` are documented here.
This project follows [Semantic Versioning](https://semver.org).

## [Unreleased] — backends & docs (no engine/published-crate change)

GPU-backend work only (the `metal/` and `cuda/` crates + `kernels/`); the published
`vitni-tensor` engine crate is unchanged since 0.3.0.

- **CUDA**: wide-tile WMMA + `mma.sync m16n8k32` + `cp.async` Q4_K GEMM (~3.4× the dp4a
  kernel), wired into the batched prefill; batched attention/RoPE/head/Q6_K. See
  [`docs/cuda-benchmark.md`](docs/cuda-benchmark.md).
- **Metal**: dtype-gap fix — Q8_0/Q5_0 linears, NeoX RoPE + QKV bias, host-side embedding —
  so Qwen2.5 (and any Q8_0/Q5_0-mixed model) runs on Metal bit-identically. regime-3 added to
  Metal, so **both regimes are now cross-vendor** (CPU / Apple Metal / NVIDIA CUDA), validated
  on an M3 Max: Qwen2.5-0.5B, TinyLlama, Mistral-7B all bit-identical to the CPU. See
  [`docs/regimes.md`](docs/regimes.md).

## [0.3.0] — 2026-10-18

**GPU backends + a second numeric regime. No published digest moves: regime-2 is unchanged and
every addition is either bit-identical to the published anchors or explicitly reserved.** New
public API (`run_quantized_regime3`, `run_quantized_with_prefill`) → minor version bump.

- **NVIDIA CUDA backend** (`cuda/` crate, `vitni-tensor-cuda`) — the full quantized forward on an
  NVIDIA GPU, **bit-identical to the CPU engine** (regime-2): validated cert-for-cert on
  Mistral-7B, TinyLlama-1.1B, and Qwen2.5-0.5B/32B across Tesla T4 (sm_75) and A10G (sm_86). With
  the existing Metal backend this extends the determinism claim across GPU vendors and CPU↔GPU.
- **Batched prefill forward** — processes all `P` prompt tokens together (weights staged once;
  batched attention/RoPE/bias; output head computed for the last position only; tiled Q6_K GEMM).
  Bit-identical certificate to the per-token path; large prefill speedups measured on an A10G
  (TinyLlama 26.9×, Qwen2.5-7B 24.9× vs per-token).
- **`regime-3` (reserved, opt-in)** — the k-quant linears (Q4_K/Q6_K) in exact integers with int8
  activations and a fixed-order running-sum combine. Deterministic and bit-identical CPU↔CUDA; it
  is the path that runs on integer **tensor cores** (where float cannot be deterministic). GPU
  kernel progression, all bit-identical: wide-WMMA → `mma.sync m16n8k32` → `cp.async`
  double-buffer (up to ~3.4× the `__dp4a` kernel on an A10G). **Reserved:** the bound `REGIME`
  string is still `vitni-regime-2` and there are no regime-3 anchors — promoting it is a
  deliberate, versioned re-anchor. Entry points `inference::run_quantized_regime3` and the
  backend's `new_regime(…, 3)`. See [`docs/regimes.md`](docs/regimes.md).
- **Fixed** (batched prefill): the dtype dispatch treated the weight kind as a boolean, routing
  Q8_0/Q5_0 tensors (e.g. Qwen's output head) to the Q6_K kernel and corrupting them — surfaced
  only on models with a non-Q4_K/Q6_K linear (Qwen2.5). Now dispatches the real dtype.
- **Engine:** `run_quantized_regime3` (CPU regime-3 reference) and `run_quantized_with_prefill`
  (batched-prefill driver), both certificate-transparent.
- **Docs:** [`docs/regimes.md`](docs/regimes.md) (the two-regime model + when to use each) and
  [`docs/cuda-benchmark.md`](docs/cuda-benchmark.md) (GPU determinism validation + throughput).

## [0.2.2] — 2026-10-16

**Fix (`std-parallel`): Q4_K ran a different, lossy kernel than the default build.**
The `std-parallel` forward computed Q4_K linears via the int8-activation *integer*
kernel instead of the f32 *fused* kernel the default build uses; the two are not
bit-identical, so an enabled-`std-parallel` build produced different certified digests
for Mistral-7B / Qwen-0.5B (TinyLlama agreed by luck and masked it). It now
parallelizes the SAME fused kernel (`linear_q4_k_fused_parallel`), bit-identical to
serial at every thread count (asserted by `parallel_q4k_matches_serial_at_every_thread_count`).
The int8 integer path stays in the tree as the separate regime-4 candidate, no longer
wired into the default parallel forward.

- **Anchor guard (`tools/check-anchors.sh` + CI).** Re-derives the published
  `regime-manifest.json` anchors from the engine and fails on drift, run under both the
  default and `--features std-parallel` builds — so a build that computes different bits
  can no longer ship silently.

**REGIME BUMP `vitni-regime-1` → `vitni-regime-2`: one reduction contract for the
whole forward pass.** `rms_norm`, `softmax`, and attention (Q·K, A·V) computed
their reductions with serial accumulators — a fixed order, but one that pinned
each row to a single thread and was a *different* shape from the canonical
lane+tree reduction the matmul already used. They now all route through the ONE
canonical reduction (`ops::quant::canonical_dot` / new `canonical_sum`): the same
`i % 8` lane assignment + fixed pairwise tree, bit-identical across thread count
and vector width, and parallelizable. This changed the forward-pass bits, so
every model digest moved and the receipt `REGIME` tag bumped — a v2 receipt
issued under regime-1 is now cryptographically distinguishable from one under
regime-2 (the tag's whole purpose). Generated tokens did NOT move (the change is
below the argmax margin), so behavior is unchanged.

- **`ops::quant::canonical_sum`** (+ `canonical_sum_pub`): the canonical reduction
  for a plain `Σ x[i]`, bit-identical to `canonical_dot(x, [1.0; n])` — proven by
  `canonical_sum_equals_dot_with_ones`. Softmax denominators use it.
- **The matmul and transcendental pins did NOT move** (`0x8a428433686d13af` and the
  transcendental hash): those reductions were already canonical. Only the three
  serial ops changed.
- **GPU parity re-verified:** `kernels/forward_ops.metal` (`rms_kernel`,
  `softmax_kernel`, `attention`) rewritten to the same lane+tree shape; the full
  TinyLlama forward on Apple GPU reproduces the CPU certificate digest under
  regime-2 (`metal_certificate_matches_cpu`).
- **`src/bin/gen-anchors.rs` + `regime-manifest.json`:** a single-source-of-truth
  generator that recomputes every anchor (matmul pin, per-model digests) in one
  command — so a future regime change is a regenerate, not a hand-sync across
  tests, spec, and paper.

**Deterministic GPU matmul, proven — the reproducible path is no longer CPU-only.**
The `SYS_GPU_MATMUL` seam existed (`accel::Accelerator`) but had no conforming
kernel to point a GPU impl at. Added the reference kernels and their proof.

- **`kernels/canonical_matmul.metal` + `kernels/canonical_matmul.cu`:** the
  canonical reduction on-GPU — fixed order (element -> lane by `i % 8`, fixed
  pairwise tree) and NO fma contraction (`fastMathEnabled=false` /
  `--fmad=false`, plus `__fmul_rn`/`__fadd_rn`).
- **Verified bit-for-bit** against the CPU reduction on real hardware: Apple M3
  Max (Metal) and NVIDIA Tesla T4 (CUDA 13.2) both reproduce the matmul pin
  `0x8a428433686d13af`, max ULP 0 across the shape sweep. Controls confirm a
  fused kernel diverges (up to 448 ULP). See `kernels/README.md` and
  `kernels/RESULTS-nvidia-t4.txt`.
- **`tests/gpu_kernel_contract.rs`:** portable CI guard that locks the kernels'
  algorithm to `canonical_dot` and the pin — no GPU required.
- Docs in `accel::mod` and `ops::matmul` now point at the reference kernels.
- No change to any digest or regime.

## [0.2.1] — 2026-08-20

Hygiene (engine-review finding 5); no change to any digest.

- **Removed dead code carrying a numerical contract:** the unused `regime2` kernel
  chain (`linear_q4_k_fused_regime2`, `canonical_dot_q4k_fused_regime2`) — 0 callers,
  nothing pinning it. The live fused kernels (pinned by `fused_q4k_dot_is_bit_identical`)
  are untouched; `canonical_dot_regime2` is scoped to `#[cfg(test)]`.
- **Genericized dangling references to a private parent codebase** in comments, docs,
  and user-facing error strings. `ExCert`, `SYS_GPU_*`, and the "Apple M3 Max" chip
  name are legitimate and preserved.

## [0.2.0] — 2026-08-20

**Tier-1 v2: the model-computation digest now binds the numerical regime.** The
certificate had no recorded arithmetic version, so a deliberate reduction change would
break level-2 replay for every prior receipt with no way to tell "regime moved" from
"tampered".

- Added `REGIME = "vitni-regime-1"`, bound into a new `"vitnify-receipt v2\x00"` digest
  domain (length-prefixed after the separator). Bump it whenever a pinned reduction hash
  moves.
- `compute_digest()` emits tier-1 **v2**; `compute_digest_v1()` retained frozen, so
  pre-regime receipts — including the anchor `9c0754…` — stay reproducible. Verified:
  the reference run reproduces v1 exactly and yields v2 `ffebe862…9c88f`.
- Pinned v2 format-anchor test (runs cross-ISA in CI); `vitni-receipt` binary emits
  `model_digest` (v2), `regime`, and `model_digest_v1`.
- Determinism tests now assert their pinned hashes (were tautologies); CI runs on
  x86-64 **and** aarch64. Cross-vendor v2 bit-identity confirmed on Apple/Graviton/Intel/AMD.

## [0.1.0] — 2026-08-19

Initial public release. Deterministic, `no_std` tensor engine for cross-vendor
bit-identical LLM inference — the engine that produces the model-computation digest
a `vitnify-receipt` binds. Pinned-order fp32 reductions and exact-integer k-quant
dots give the same output bits across CPU vendors and instruction sets; the
`vitni-receipt` binary emits the tier-1 digest (conformance anchor
`9c0754458633e863e0fb5bb2bd00df0d8b813934687b9a4097a1a9a4179f3b0f`) for a run.
