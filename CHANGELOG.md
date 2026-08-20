# Changelog

All notable changes to `vitni-tensor` are documented here.
This project follows [Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-08-19

Initial public release. Deterministic, `no_std` tensor engine for cross-vendor
bit-identical LLM inference — the engine that produces the model-computation digest
a `vitnify-receipt` binds. Pinned-order fp32 reductions and exact-integer k-quant
dots give the same output bits across CPU vendors and instruction sets; the
`vitni-receipt` binary emits the tier-1 digest (conformance anchor
`9c0754458633e863e0fb5bb2bd00df0d8b813934687b9a4097a1a9a4179f3b0f`) for a run.
