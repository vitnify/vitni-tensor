//! ExCert — Execution Certificate.
//!
//! A cert binds a set of named inputs and named outputs into a single
//! BLAKE3 digest. The cryptographic claim it supports:
//!
//!   "These specific output bytes were produced by some deterministic
//!    process running over these specific input bytes."
//!
//! Verification = re-run the same process on the same inputs, compute
//! the cert hash, check it matches. Because vitni-tensor's forward
//! pass is bit-for-bit deterministic (proven in M3), this claim is
//! meaningful — a verifier with the same Build ID, weights, and
//! prompt can independently confirm the output.
//!
//! On the host runtime the substrate ALSO signs the cert with its kernel key so
//! verifiers can trust the cert was produced by an authentic process.
//! That signing layer is separate from the
//! binding layer in this module; both produce the same binding
//! digest, so software-only cert generation (host tests) cross-
//! verifies certificates independently, without trusting any runtime.
//!
//! ## Binding format
//!
//! A cert is `(inputs, outputs, digest)` where:
//!
//!   `digest = BLAKE3(`
//!     `"vitnify-receipt v2" || 0x00 ||`
//!     `LEB128(|regime|) || regime ||`
//!     `LEB128(n_inputs)  || for each: LEB128(|name|) || name || LEB128(|bytes|) || bytes ||`
//!     `LEB128(n_outputs) || for each: same format`
//!   `)`
//!
//! `regime` (see [`builder::REGIME`]) names the numerical reduction contract the
//! forward pass ran under. Binding it is what lets a verifier tell a deliberate regime
//! change ("cannot replay under this engine") from tampering; v1 (frozen) omitted it.
//!
//! What the SHIPPED `vitni-receipt` binary commits is an **I/O pair under a fixed
//! engine**: the inputs above are `(model_id, weights_hash, arch_hash, prompt_tokens,
//! n_new_tokens)` and the outputs are `(output_tokens, output_tokens_hash)`, with
//! `n_ops = 0`. Per-op / activation records are an available mode (`ExCertMode::PerOp`,
//! well tested) that the shipped binary does not emit — so a different implementation
//! producing the same tokens under the same regime yields the same digest.
//!
//! Names and byte-blobs are length-prefixed so concatenation is
//! injective (no `(["a", "bc"], "")` vs `(["ab", "c"], "")` collision).
//! Inputs are bound BEFORE outputs are folded in so the cert hash is
//! computable streaming — a verifier processing inputs first knows the
//! "pre-output" digest matches before doing the expensive output
//! recomputation.

pub mod builder;
pub mod sink;

pub use builder::{ActivationRecord, Cert, CertBuilder, CertField, InterventionRecord, OpRecord};
pub use sink::{CertSink, NullSink, RecordingSink, SinkEvent};
