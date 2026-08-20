# Changelog

All notable changes to `vitni-tensor` are documented here.
This project follows [Semantic Versioning](https://semver.org).

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
