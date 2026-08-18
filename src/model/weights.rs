//! Weight views over a llama2.c-format binary blob.
//!
//! No copies — every field is a slice into the blob. The blob is
//! expected to live as long as the `Weights` struct (typical pattern:
//! `include_bytes!()` on the target runtime, mmap from P3 partition in production).
//!
//! Layout, in order, exactly matches karpathy's `llama2.c`:
//!   token_embedding_table  (vocab × dim)
//!   rms_att_weight         (n_layers × dim)
//!   wq                     (n_layers × dim × (n_heads × head_size))
//!   wk                     (n_layers × dim × (n_kv_heads × head_size))
//!   wv                     (n_layers × dim × (n_kv_heads × head_size))
//!   wo                     (n_layers × (n_heads × head_size) × dim)
//!   rms_ffn_weight         (n_layers × dim)
//!   w1                     (n_layers × hidden_dim × dim)
//!   w2                     (n_layers × dim × hidden_dim)
//!   w3                     (n_layers × hidden_dim × dim)
//!   rms_final_weight       (dim,)
//!   [freq_cis_real]        (seq_len × head_size / 2)   — skipped (RoPE computed live)
//!   [freq_cis_imag]        (seq_len × head_size / 2)   — skipped
//!   wcls                   (vocab × dim)               — absent if shared_weights

use super::config::Config;
use crate::error::{Error, Result};

/// Borrowed weight views into a llama2.c binary blob.
#[derive(Debug)]
pub struct Weights<'a> {
    /// Token embedding table — `[vocab, dim]`.
    pub token_embedding_table: &'a [f32],
    /// Per-layer attention RMSNorm weights — `[n_layers, dim]`.
    pub rms_att_weight: &'a [f32],
    /// Per-layer Q projection — `[n_layers, dim, n_heads * head_size]`.
    pub wq: &'a [f32],
    /// Per-layer K projection — `[n_layers, dim, n_kv_heads * head_size]`.
    pub wk: &'a [f32],
    /// Per-layer V projection — `[n_layers, dim, n_kv_heads * head_size]`.
    pub wv: &'a [f32],
    /// Per-layer output projection — `[n_layers, n_heads * head_size, dim]`.
    pub wo: &'a [f32],
    /// Per-layer FFN RMSNorm weights — `[n_layers, dim]`.
    pub rms_ffn_weight: &'a [f32],
    /// Per-layer FFN gate matrix — `[n_layers, hidden_dim, dim]`.
    pub w1: &'a [f32],
    /// Per-layer FFN down matrix — `[n_layers, dim, hidden_dim]`.
    pub w2: &'a [f32],
    /// Per-layer FFN up matrix — `[n_layers, hidden_dim, dim]`.
    pub w3: &'a [f32],
    /// Final RMSNorm before lm_head — `[dim]`.
    pub rms_final_weight: &'a [f32],
    /// LM head — `[vocab, dim]`. Aliases `token_embedding_table` when
    /// `cfg.shared_weights` is true.
    pub wcls: &'a [f32],
}

impl<'a> Weights<'a> {
    /// Carve a `Weights` view from a llama2.c-format blob.
    ///
    /// The blob's first 28 bytes are the config header (parse separately
    /// via `Config::from_header`); the remainder is f32 weights in the
    /// layout documented in the module header.
    pub fn from_blob(blob: &'a [u8], cfg: &Config) -> Result<Self> {
        const HEADER_BYTES: usize = 28;
        if blob.len() < HEADER_BYTES {
            return Err(Error::InvalidShape("weights blob shorter than header"));
        }
        let weights_bytes = &blob[HEADER_BYTES..];
        if weights_bytes.len() % 4 != 0 {
            return Err(Error::InvalidShape("weights blob not 4-byte aligned"));
        }
        // SAFETY: f32 has 4-byte alignment requirement; the blob slice
        // points at the start of the weights region which is f32-aligned
        // by file-format contract. Bounds checked above.
        let weights_f32: &[f32] = unsafe {
            core::slice::from_raw_parts(
                weights_bytes.as_ptr() as *const f32,
                weights_bytes.len() / 4,
            )
        };

        let head_size = cfg.head_size();
        // Position cursor (in f32 elements) + take helper as plain
        // mutation. Inlining instead of a closure so the borrow
        // checker can prove non-overlap of the returned slices.
        let mut p: usize = 0;
        fn take<'a>(src: &'a [f32], p: &mut usize, len: usize) -> Result<&'a [f32]> {
            if *p + len > src.len() {
                return Err(Error::InvalidShape("weights blob truncated"));
            }
            let s = &src[*p..*p + len];
            *p += len;
            Ok(s)
        }

        let token_embedding_table = take(weights_f32, &mut p, cfg.vocab_size * cfg.dim)?;
        let rms_att_weight = take(weights_f32, &mut p, cfg.n_layers * cfg.dim)?;
        let wq = take(
            weights_f32,
            &mut p,
            cfg.n_layers * cfg.dim * (cfg.n_heads * head_size),
        )?;
        let wk = take(
            weights_f32,
            &mut p,
            cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size),
        )?;
        let wv = take(
            weights_f32,
            &mut p,
            cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size),
        )?;
        let wo = take(
            weights_f32,
            &mut p,
            cfg.n_layers * (cfg.n_heads * head_size) * cfg.dim,
        )?;
        let rms_ffn_weight = take(weights_f32, &mut p, cfg.n_layers * cfg.dim)?;
        let w1 = take(
            weights_f32,
            &mut p,
            cfg.n_layers * cfg.hidden_dim * cfg.dim,
        )?;
        let w2 = take(
            weights_f32,
            &mut p,
            cfg.n_layers * cfg.dim * cfg.hidden_dim,
        )?;
        let w3 = take(
            weights_f32,
            &mut p,
            cfg.n_layers * cfg.hidden_dim * cfg.dim,
        )?;
        let rms_final_weight = take(weights_f32, &mut p, cfg.dim)?;
        // Skip legacy RoPE freq tables (we compute on the fly in `forward`).
        p += cfg.seq_len * head_size / 2;
        p += cfg.seq_len * head_size / 2;
        let wcls = if cfg.shared_weights {
            token_embedding_table
        } else {
            take(weights_f32, &mut p, cfg.vocab_size * cfg.dim)?
        };
        let _ = p; // suppress unused if shared_weights branch doesn't read it

        Ok(Self {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }
}
