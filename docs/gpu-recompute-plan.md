# Shipping GPU recompute

## Why

Level-2 recompute (the dispute path: re-run the model and confirm the committed
trajectory) runs on CPU today. The deterministic GPU forward already exists and is
bit-exact: the Metal backend issues a certificate whose digest is identical to the CPU's,
validated on this hardware (`metal_certificate_matches_cpu` passes; shared digest confirmed
by `metal/tax-matrix.sh`). Wiring it into the shipped recompute path makes dispute-path
recompute meaningfully faster with zero change to the certificate.

> **Update (correctness, found while testing this plan):** `MetalForward` diverged from
> CPU on **Mistral-7B** — its shortcut of pre-dequantizing Q4_K to f32 and running
> `f32_matvec` was not bit-safe at 7B matrix sizes (TinyLlama's smaller dims agreed by
> luck). Fixed by switching to the fused `q4k_linear` kernel (proven bit-identical in
> `kernels/conformance-forward`); now verified bit-exact on both TinyLlama and Mistral-7B,
> and guarded by `metal_certificate_matches_cpu_7b`. The throughput table below predates
> the fix; post-fix Mistral-7B GPU is ~5.2 tok/s (bit-exact), ~8x over deterministic CPU.
> A bit-safe pre-dequant to recover the lost speed is a later optimization.

## Measured tax (reproduce with `metal/tax-matrix.sh`)

TinyLlama-1.1B Q4_K_M, Apple M-series, token-at-a-time decode:

| backend                              | tok/s  | vs native (same device) |
|--------------------------------------|--------|-------------------------|
| native llama.cpp, CPU (10 thr, BLAS) | 118.75 | 1.0x                    |
| native llama.cpp, Metal GPU          | 252.68 | 1.0x                    |
| deterministic CPU (serial)           | 4.1    | ~29x slower than native CPU |
| deterministic CPU (std-parallel)     | 16.8   | ~7x slower than native CPU  |
| deterministic GPU (Metal)            | 29.4   | ~8.6x slower than native Metal |

All three deterministic backends produce a BIT-IDENTICAL certificate (same digest). So the
determinism tax is roughly 7-9x versus native on the same device, and the GPU is the fastest
deterministic option (7.2x over serial CPU, 1.75x over parallel CPU). (7B numbers: see the
harness output; the GPU advantage grows with model size.)

## Current state (built vs not)

- **CPU core** — `vitni-tensor` (crates.io), `no_std`. The `inference::run_quantized_with_forward`
  seam decouples the forward from the engine, which is what makes a GPU backend a drop-in.
- **Metal** — `vitni-tensor-metal` crate: the full quantized forward on Apple GPU, bit-exact,
  VALIDATED here. It is a separate std/macOS workspace and is NOT published.
- **CUDA** — a full-forward conformance kernel (`kernels/cuda_forward_conformance.cu`) proves
  bit-identity on NVIDIA (T4 run in `kernels/RESULTS-nvidia-t4.txt`), but there is NO host
  crate wiring it into inference (no `cuda/` equivalent of `metal/`).
- **Shipped recompute** — the `vitni-receipt` binary (`src/bin/vitni-receipt.rs`) is CPU-only,
  no device flag. The SDK's Level-2 shells out to it (`VITNI_RECEIPT_BIN`).

## What shipping requires

1. **Backend selection** in the recompute entry point: `vitni-receipt --device cpu|metal|cuda`
   (auto-detect, fall back to CPU). On macOS, link `vitni-tensor-metal` behind a `metal` feature.
2. **Packaging / distribution.** The GPU backends are std + platform-specific, so they cannot
   live in the `no_std` crates.io lib. Options: prebuilt per-platform `vitni-receipt` binaries
   (GitHub Releases), or `cargo install vitni-receipt --features metal` (macOS) / `--features cuda`
   (needs a CUDA toolkit). The SDK's `VITNI_RECEIPT_BIN` points at whichever is installed.
3. **SDK** — minimal. Level-2 already shells out; it should prefer a GPU-capable binary and pass
   `--device auto`. Surface the backend used, for transparency in the verdict.
4. **CI determinism gate (non-negotiable).** Run the GPU bit-identity conformance on GPU runners
   (Apple Metal + an NVIDIA runner, as the T4 run did) on every change to the kernels. The T4
   CONTROL shows a fused build diverges, so a kernel or build-flag regression could silently move
   the bits. The gate is what keeps "GPU recompute" honest.

## Phased plan

- **Phase 1 — Metal (proven, least new code).** Feature-gate `vitni-receipt` to link
  `vitni-tensor-metal` on macOS; auto-use the GPU, fall back to CPU; wire the SDK to use it; add
  an Apple-GPU conformance CI job; ship a prebuilt macOS-arm64 binary. The hard part (the bit-exact
  Metal forward) is DONE and validated here.
- **Phase 2 — CUDA (more work).** Build a `vitni-tensor-cuda` host crate mirroring `metal/`,
  wrapping the forward kernels the conformance `.cu` already proves bit-exact; add `--device cuda`;
  add an NVIDIA conformance CI job (the T4 run is the template); ship a prebuilt linux-cuda binary.
- **Phase 3 — distribution / UX.** Prebuilt per-platform binaries via GitHub Releases; have
  `pip install vitnify` detect and point `VITNI_RECEIPT_BIN` at the right one, so a user gets fast
  recompute without installing a GPU toolchain.

## Risks / open questions

- The deterministic GPU kernel is conservative (one command buffer per token, no fast math, fixed
  reduction order), so it lands ~8.6x off native Metal. It is the fastest deterministic option, but
  there is headroom to optimize the GPU kernel WITHIN the fixed reduction contract (a separate perf
  effort, not a blocker for shipping).
- The CUDA host crate is net-new code plus a CUDA build in CI, heavier than Metal.
- Distributing platform-specific binaries is release engineering, separate from the core work.

## Recommendation

Do **Phase 1 (Metal) first**: it is validated, low-risk, and covers Mac dev/CI plus Apple-silicon
deployments, gated behind the CI conformance job. Add **Phase 2 (CUDA)** when there is demand for
Linux/NVIDIA recompute at scale. Treat GPU-kernel speed optimization (closing the ~8.6x) as a
later, separate effort.
