//! Llama2 model — the first end-to-end demonstration that ops in
//! `src/ops/*` compose into a real, runnable LLM architecture.
//!
//! Architecture follows karpathy's llama2.c (also what `the reference implementation`
//! runs) so we have a known-good reference for cross-verification.
//!
//! ## Layout
//!
//! - `config` — model dimensions parsed from a llama2.c-format header
//! - `weights` — typed views over the binary weight blob, no copies
//! - `forward` — single-token forward pass returning logits, plus
//!   greedy-decode driver
//!
//! ## Status (M3)
//!
//! - Architecture defined; compiles clean no_std
//! - Verified bit-identical against an in-test reference impl on
//!   synthetic weights matching stories15M's shape
//! - Real stories15M weight load is the next plumbing step (host-side
//!   test reads the asset blob, runtime-side reads from P3 partition)

pub mod config;
pub mod forward;
pub mod forward_quantized;
pub mod gemma;
pub mod gguf;
pub mod inference;
pub mod quant_weights;
pub mod weights;
