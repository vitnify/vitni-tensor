# Changelog

All notable changes to `vitni-tensor` are documented here.
This project follows [Semantic Versioning](https://semver.org).

## [Unreleased]

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
