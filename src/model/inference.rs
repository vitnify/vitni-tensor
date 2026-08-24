//! High-level inference driver: load → forward N steps → bind output
//! into an ExCert.
//!
//! Mirrors `the reference`'s cert format exactly so software-computed
//! certs (host tests) and runtime-issued certs (the host runtime)
//! produce identical binding digests for the same inputs/outputs.

use super::{
    config::Config,
    forward::{self, RunState},
    weights::Weights,
};
use crate::{
    cert::{Cert, CertBuilder, CertSink},
    error::Result,
    storage::Storage,
    tensor::Tensor,
};
use alloc::{format, string::String, vec::Vec};

/// Inference request: the inputs the cert will bind to.
pub struct Request<'a> {
    /// Model architecture identifier — short string id like
    /// `"stories15M-llama2c"`. Doesn't change per inference; binds the
    /// cert to a specific model family.
    pub model_id: &'a str,
    /// Initial token IDs (the prompt). For the cert we include the
    /// prompt bytes themselves (small).
    pub prompt_tokens: &'a [u32],
    /// Number of NEW tokens to generate beyond the prompt.
    pub n_new_tokens: usize,
}

/// Inference output: generated tokens plus the cert binding them to
/// the inputs.
pub struct Outcome {
    /// All tokens generated (the new ones — does NOT include the prompt).
    pub generated_tokens: Vec<u32>,
    /// Execution certificate.
    pub cert: Cert,
}

/// Canonical architecture string used as the `arch_hash` input.
/// Format matches `the reference`'s `Config::arch_string`.
pub fn arch_string(cfg: &Config) -> String {
    format!(
        "llama2c:dim={},hidden={},layers={},heads={},kv_heads={},vocab={},seq={},shared={}",
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.vocab_size,
        cfg.seq_len,
        cfg.shared_weights,
    )
}

/// Run a full prompt+generate cycle and emit an ExCert.
///
/// Cert fields (matching `the reference`):
///
///   inputs:
///     - model_id        (utf-8 bytes)
///     - weights_hash    (32 bytes, BLAKE3 of the full weights blob)
///     - arch_hash       (32 bytes, BLAKE3 of `arch_string(cfg)`)
///     - prompt_tokens   (4 × len bytes, U32 LE token IDs)
///     - n_new_tokens    (4 bytes, U32 LE)
///
///   outputs:
///     - output_tokens   (4 × len bytes, U32 LE generated token IDs)
///     - output_tokens_hash (32 bytes, BLAKE3 of output_tokens for
///                          quick digest comparison without unpacking)
///
/// Note: `weights_hash` is required from the caller, not computed
/// here, because the framework only sees the parsed `Weights` views,
/// not the original blob. The caller (which loaded the blob from
/// disk / CAS / include_bytes!) is the right place to compute it.
pub fn run(
    cfg: &Config,
    weights: &Weights,
    weights_blob_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome> {
    run_with_sink(
        cfg,
        weights,
        weights_blob_hash,
        req,
        &mut crate::cert::NullSink,
        0,
    )
    .map(|(outcome, _id)| outcome)
    .map_err(|e| match e {
        RunError::Inference(e) => e,
        RunError::Sink(_) => unreachable!("NullSink is infallible"),
    })
}

/// Inference + ExCert with runtime submission.
///
/// Identical to `run` except every cert declaration is also replayed
/// into `sink`. On the host runtime with a `RuntimeSink`, this issues a real
/// kernel cert; the returned software `Cert.digest` equals the
/// kernel-issued digest by construction.
///
/// Returns `(Outcome, substrate_cert_id)` on success; the second value
/// is whatever the sink's `on_finalize` returned (0 for `NullSink`,
/// the real kernel cert id for `RuntimeSink`).
pub fn run_with_sink<S: CertSink>(
    cfg: &Config,
    weights: &Weights,
    weights_blob_hash: &[u8; 32],
    req: &Request,
    sink: &mut S,
    tier: u8,
) -> core::result::Result<(Outcome, u64), RunError<S::Error>> {
    // ---- collect inputs into the cert ----
    let mut builder = CertBuilder::new();
    builder.declare_input("model_id", req.model_id.as_bytes());
    builder.declare_input("weights_hash", weights_blob_hash);

    let arch_str = arch_string(cfg);
    let arch_hash = *::blake3::hash(arch_str.as_bytes()).as_bytes();
    builder.declare_input("arch_hash", &arch_hash);

    let prompt_bytes = u32_slice_to_le_bytes(req.prompt_tokens);
    builder.declare_input("prompt_tokens", &prompt_bytes);

    let n_new_bytes = (req.n_new_tokens as u32).to_le_bytes();
    builder.declare_input("n_new_tokens", &n_new_bytes);

    // ---- run the inference ----
    let mut state = RunState::new(cfg);
    let mut generated: Vec<u32> = Vec::with_capacity(req.n_new_tokens);

    // Decode loop mirrors the reference/karpathy llama2.c:
    //
    //   for pos in 0..(prompt_len + n_new):
    //     logits = forward(cur, pos)
    //     next = if pos < prompt_len - 1: prompt[pos+1]     (still feeding prompt)
    //            else:                    argmax(logits)
    //     if pos >= prompt_len: push(next)                  (past prompt — new token)
    //     cur = next
    //
    // The split lets pos == prompt_len-1 compute the LAST prompt-driven
    // forward pass (which populates KV cache for that position) without
    // either forcing OR pushing — its argmax becomes the input at
    // pos = prompt_len, where pushing starts.
    let prompt_len = req.prompt_tokens.len();
    if prompt_len == 0 {
        return Err(RunError::Inference(crate::error::Error::InvalidShape(
            "prompt_tokens must not be empty",
        )));
    }
    let total = prompt_len + req.n_new_tokens;
    if total > cfg.seq_len {
        return Err(RunError::Inference(crate::error::Error::InvalidShape(
            "prompt + n_new_tokens exceeds seq_len",
        )));
    }
    let mut cur = req.prompt_tokens[0];
    for pos in 0..total {
        let logits = forward::step(cfg, weights, &mut state, cur, pos)
            .map_err(RunError::Inference)?;
        let argmax = logits.argmax_last_dim().map_err(RunError::Inference)?;
        let picked = argmax_u32(&argmax);
        let next = if pos + 1 < prompt_len {
            req.prompt_tokens[pos + 1]
        } else {
            picked
        };
        if pos >= prompt_len {
            generated.push(next);
        }
        cur = next;
    }

    // ---- bind outputs ----
    let output_bytes = u32_slice_to_le_bytes(&generated);
    let output_hash = *::blake3::hash(&output_bytes).as_bytes();
    builder.declare_output("output_tokens", &output_bytes);
    builder.declare_output("output_tokens_hash", &output_hash);

    // Finalize: replays every declaration into the sink, then computes
    // the software digest. On the host runtime the sink issues the real kernel
    // cert here; the digest the kernel computes equals the one we
    // return because both bind the same length-prefixed pairs.
    let (cert, cert_id) = builder
        .finalize_with_sink(sink, tier)
        .map_err(RunError::Sink)?;

    Ok((
        Outcome {
            generated_tokens: generated,
            cert,
        },
        cert_id,
    ))
}

/// Composite error from `run_with_sink`: either the inference itself
/// failed (shape mismatch, op error) or the sink rejected a call.
#[derive(Debug)]
pub enum RunError<S> {
    /// Forward pass / tensor op failure.
    Inference(crate::error::Error),
    /// Runtime sink rejected a call (e.g. `the runtime's cert syscall` failed).
    Sink(S),
}

fn u32_slice_to_le_bytes(ts: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ts.len() * 4);
    for &t in ts {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

fn argmax_u32(t: &Tensor) -> u32 {
    let Storage::Cpu(s) = t.storage() else {
        panic!("argmax tensor not on CPU");
    };
    u32::from_le_bytes(s.as_bytes()[..4].try_into().unwrap())
}

// ============================================================================
// Quantized inference — parallel to `run` / `run_with_sink` but takes a
// `QuantizedWeights` (from a parsed GGUF) and dispatches into
// `forward_quantized::step` which routes projections through
// `linear_q4_0` instead of plain f32 matmul.
//
// Cert binding is identical to the F32 path so a runtime-issued
// receipt for a Q4_0 run can be cross-verified against a reference
// implementation that dequantizes the same GGUF and runs the F32
// forward — the inputs/outputs hash the same byte sequences.
// ============================================================================

use super::forward_quantized;
use super::quant_weights::QuantizedWeights;

/// Run a full prompt+generate cycle on Q4_0 / Q8_0 / mixed-dtype
/// quantized weights and emit an ExCert. CPU-only convenience
/// wrapper around `run_quantized_with_sink` with `NullSink`.
pub fn run_quantized(
    cfg: &Config,
    weights: &QuantizedWeights<'_>,
    weights_blob_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome> {
    run_quantized_with_sink(
        cfg,
        weights,
        weights_blob_hash,
        req,
        &mut crate::cert::NullSink,
        0,
    )
    .map(|(o, _)| o)
    .map_err(|e| match e {
        RunError::Inference(e) => e,
        RunError::Sink(_) => unreachable!("NullSink is infallible"),
    })
}

/// Quantized inference + runtime-issued ExCert.
///
/// Bound to the SAME cert input/output schema as `run_with_sink`:
/// model_id, weights_hash, arch_hash, prompt_tokens, n_new_tokens →
/// output_tokens, output_tokens_hash. A verifier with the GGUF
/// blob can recompute weights_hash + arch_hash and recheck the cert
/// against a reference implementation (llama.cpp, etc.) — the
/// digests must match because both sides bind the same bytes.
pub fn run_quantized_with_sink<S: CertSink>(
    cfg: &Config,
    weights: &QuantizedWeights<'_>,
    weights_blob_hash: &[u8; 32],
    req: &Request,
    sink: &mut S,
    tier: u8,
) -> core::result::Result<(Outcome, u64), RunError<S::Error>> {
    let mut state = RunState::new(cfg);
    run_quantized_with_forward(cfg, weights_blob_hash, req, sink, tier, move |cur, pos| {
        forward_quantized::step(cfg, weights, &mut state, cur, pos)
    })
}

/// The cert-issuing generation loop, parameterized on the per-step forward.
///
/// The CPU path (`run_quantized_with_sink`) passes `forward_quantized::step`; a
/// GPU backend (a separate host crate — the `no_std` engine can't link Metal or
/// CUDA) passes its own step producing the same logits, so the certificate is
/// **identical** (it binds inputs + output tokens, both backend-independent).
/// `forward(token, pos)` returns the logits for `token` at that position; the
/// KV-cache state belongs to the closure.
pub fn run_quantized_with_forward<S, F>(
    cfg: &Config,
    weights_blob_hash: &[u8; 32],
    req: &Request,
    sink: &mut S,
    tier: u8,
    mut forward: F,
) -> core::result::Result<(Outcome, u64), RunError<S::Error>>
where
    S: CertSink,
    F: FnMut(u32, usize) -> crate::error::Result<Tensor>,
{
    let mut builder = CertBuilder::new();
    builder.declare_input("model_id", req.model_id.as_bytes());
    builder.declare_input("weights_hash", weights_blob_hash);

    let arch_str = arch_string(cfg);
    let arch_hash = *::blake3::hash(arch_str.as_bytes()).as_bytes();
    builder.declare_input("arch_hash", &arch_hash);

    let prompt_bytes = u32_slice_to_le_bytes(req.prompt_tokens);
    builder.declare_input("prompt_tokens", &prompt_bytes);

    let n_new_bytes = (req.n_new_tokens as u32).to_le_bytes();
    builder.declare_input("n_new_tokens", &n_new_bytes);

    let mut generated: Vec<u32> = Vec::with_capacity(req.n_new_tokens);
    let prompt_len = req.prompt_tokens.len();
    if prompt_len == 0 {
        return Err(RunError::Inference(crate::error::Error::InvalidShape(
            "prompt_tokens must not be empty",
        )));
    }
    let total = prompt_len + req.n_new_tokens;
    if total > cfg.seq_len {
        return Err(RunError::Inference(crate::error::Error::InvalidShape(
            "prompt + n_new_tokens exceeds seq_len",
        )));
    }

    let mut cur = req.prompt_tokens[0];
    for pos in 0..total {
        let logits = forward(cur, pos).map_err(RunError::Inference)?;
        let argmax = logits.argmax_last_dim().map_err(RunError::Inference)?;
        let picked = argmax_u32(&argmax);
        let next = if pos + 1 < prompt_len {
            req.prompt_tokens[pos + 1]
        } else {
            picked
        };
        if pos >= prompt_len {
            generated.push(next);
        }
        cur = next;
    }

    let output_bytes = u32_slice_to_le_bytes(&generated);
    let output_hash = *::blake3::hash(&output_bytes).as_bytes();
    builder.declare_output("output_tokens", &output_bytes);
    builder.declare_output("output_tokens_hash", &output_hash);

    let (cert, cert_id) = builder
        .finalize_with_sink(sink, tier)
        .map_err(RunError::Sink)?;

    Ok((Outcome { generated_tokens: generated, cert }, cert_id))
}
