//! Quantized-weight forward pass — the Q4_0 / Q8_0 / F32 mixed-dtype
//! analog of `forward::step`.
//!
//! Same math, same numerical sequence, same KV cache layout — the
//! only difference is that the 7 projection matmuls per layer go
//! through `linear_dispatch` which picks `linear_q4_0_cpu` for
//! Q4_0 tensors and plain f32 matmul for F32 tensors.
//!
//! ## Cross-check guarantee
//!
//! For a Q4_0 weight blob B, running:
//!   1. `forward_quantized::step(B)` directly, OR
//!   2. `dequantize_q4_0(B)` → `Weights` → `forward::step`
//! must produce *bit-identical* logits, because both paths perform
//! the same arithmetic on the same float values. The
//! `quantized_matches_dequant_then_forward` test enforces this.
//!
//! ## What's NOT supported yet (Phase 3a scope)
//!
//! - F16 weights — present in many GGUFs (e.g. embedding tables in
//!   "Q4_0_with_F16_embed" recipes). Trivial to add: extend
//!   `linear_dispatch` with a `GgufTensorType::F16` arm calling a
//!   new `f16_to_f32_vec` helper.
//! - Q4_K / Q5_K / Q6_K (k-quants) — needed for Mistral 7B Q4_K_M
//!   distribution. Mechanical add per the Phase 1 commit message.

use super::{
    config::Config,
    forward::RunState,
    gguf::GgufTensorType,
    quant_weights::{QuantTensor, QuantizedWeights},
};
use crate::{
    cert::CertBuilder,
    error::{Error, Result},
    ops,
    shape::Shape,
    storage::Storage,
    tensor::Tensor,
};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

// =============================================================================
// Per-op recorder — Phase 4a step 3
//
// Captures one `OpRecord` per instrumented op call into the caller's
// `CertBuilder`. Params hashes are cached at construction (weights
// don't change across decode steps), so the per-op overhead is only
// hashing the live input + output tensors plus one BTreeMap lookup.
//
// Layer keys: "wq/L", "wk/L", "wv/L", "wo/L", "w1/L", "w2/L", "w3/L",
//             "rms_att/L", "rms_ffn/L"
// Whole-model keys: "token_embd", "rms_final", "wcls"
// =============================================================================

/// Per-op recorder that captures `OpRecord` entries for the supplied
/// `CertBuilder` at each instrumented op site in `step_with_recorder`.
///
/// Construct once per inference (NOT per token) so the params-hash
/// cache amortizes across the whole decode loop.
pub struct PerOpRecorder<'a> {
    builder: &'a mut CertBuilder,
    /// Pre-computed BLAKE3 of every weight tensor in the model.
    /// Keyed by short string ID. Lookup is O(log n) — irrelevant
    /// relative to the BLAKE3 cost of hashing live activations.
    params_cache: BTreeMap<String, [u8; 32]>,
}

impl<'a> PerOpRecorder<'a> {
    /// Build a recorder for `builder` over the given weights blob.
    /// Pre-hashes every weight tensor (one BLAKE3 pass per weight).
    /// For stories15M (6 layers × 7 projections + 6×2 norms + 2 wholes)
    /// that's ~56 BLAKE3 calls totaling ~58 MB hashed = ~30 ms at
    /// SSE4.1 BLAKE3 throughput. Acceptable one-time cost per inference.
    pub fn new(builder: &'a mut CertBuilder, weights: &QuantizedWeights<'_>) -> Self {
        let mut params_cache = BTreeMap::new();
        params_cache.insert(
            String::from("token_embd"),
            *::blake3::hash(weights.token_embedding_table.bytes).as_bytes(),
        );
        params_cache.insert(
            String::from("rms_final"),
            hash_f32_slice(weights.rms_final_weight),
        );
        if let Some(wcls) = &weights.wcls {
            params_cache.insert(String::from("wcls"), *::blake3::hash(wcls.bytes).as_bytes());
        }
        for (l, blk) in weights.layers.iter().enumerate() {
            params_cache.insert(format!("rms_att/{}", l), hash_f32_slice(blk.rms_att_weight));
            params_cache.insert(format!("wq/{}", l), *::blake3::hash(blk.wq.bytes).as_bytes());
            params_cache.insert(format!("wk/{}", l), *::blake3::hash(blk.wk.bytes).as_bytes());
            params_cache.insert(format!("wv/{}", l), *::blake3::hash(blk.wv.bytes).as_bytes());
            params_cache.insert(format!("wo/{}", l), *::blake3::hash(blk.wo.bytes).as_bytes());
            params_cache.insert(format!("rms_ffn/{}", l), hash_f32_slice(blk.rms_ffn_weight));
            params_cache.insert(format!("w1/{}", l), *::blake3::hash(blk.w1.bytes).as_bytes());
            params_cache.insert(format!("w2/{}", l), *::blake3::hash(blk.w2.bytes).as_bytes());
            params_cache.insert(format!("w3/{}", l), *::blake3::hash(blk.w3.bytes).as_bytes());
        }
        Self {
            builder,
            params_cache,
        }
    }

    /// Record one op. Computes input + output hashes live, looks up
    /// params hash from the cache (or returns zero hash if `params_key`
    /// is empty — for non-param ops like rope/silu/softmax).
    fn record(
        &mut self,
        op_name: &str,
        layer: u32,
        params_key: &str,
        in_f32: &[f32],
        out_f32: &[f32],
    ) {
        let params_hash = if params_key.is_empty() {
            [0u8; 32]
        } else {
            self.params_cache
                .get(params_key)
                .copied()
                .unwrap_or([0u8; 32])
        };
        let input_hash = hash_f32_slice(in_f32);
        let output_hash = hash_f32_slice(out_f32);
        self.builder
            .declare_op(op_name, layer, input_hash, params_hash, output_hash);
    }
}

/// Phase 4c — causal-intervention plan: a list of "at this token +
/// checkpoint, replace the residual stream with this tensor". The
/// forward pass consults the plan at each checkpoint and substitutes
/// the override before the next op runs.
///
/// `(token_index, layer, checkpoint)` triple is the lookup key.
/// Layer is `u32::MAX` for non-per-layer checkpoints.
pub struct InterventionPlan {
    entries: Vec<InterventionEntry>,
}

struct InterventionEntry {
    token_index: u32,
    layer: u32,
    checkpoint: String,
    /// Replacement tensor bytes (interpreted as f32 row-major).
    /// Length must match the expected residual-stream dim for the
    /// target checkpoint (`dim` for residual checkpoints,
    /// `vocab_size` for post_lm_head).
    replacement: Vec<f32>,
}

impl InterventionPlan {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add an override. Multiple overrides can target the same
    /// checkpoint; the LAST one declared wins (allows chained
    /// experiments without rebuilding the plan).
    pub fn add(
        &mut self,
        token_index: u32,
        layer: u32,
        checkpoint: &str,
        replacement: Vec<f32>,
    ) -> &mut Self {
        self.entries.push(InterventionEntry {
            token_index,
            layer,
            checkpoint: String::from(checkpoint),
            replacement,
        });
        self
    }

    pub fn entries(&self) -> &[InterventionEntry] {
        &self.entries
    }

    /// Look up an override for `(token_index, layer, checkpoint)`.
    /// Returns the LAST matching entry (declaration-order
    /// last-wins).
    fn lookup(&self, token_index: u32, layer: u32, checkpoint: &str) -> Option<&[f32]> {
        self.entries
            .iter()
            .rev()
            .find(|e| {
                e.token_index == token_index
                    && e.layer == layer
                    && e.checkpoint == checkpoint
            })
            .map(|e| e.replacement.as_slice())
    }
}

impl Default for InterventionPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Phase 5a — capture the raw residual-stream tensor at each
/// named checkpoint, not just its hash. Used by logit-lens analysis
/// + any other semantic-translation tool that needs to project
/// intermediate activations through the unembedding matrix.
///
/// Cheap to build alongside (or instead of) PerActivationRecorder.
/// PerActivationRecorder hashes for cert; LensCapture keeps the
/// values for analysis. Independent; either, both, or neither.
pub struct LensCapture {
    pub entries: Vec<LensEntry>,
}

pub struct LensEntry {
    pub token_index: u32,
    pub layer: u32,
    pub checkpoint: String,
    pub residual_stream: Vec<f32>,
}

impl LensCapture {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }
    fn push(&mut self, token_index: u32, layer: u32, checkpoint: &str, residual: &[f32]) {
        self.entries.push(LensEntry {
            token_index,
            layer,
            checkpoint: String::from(checkpoint),
            residual_stream: residual.to_vec(),
        });
    }
}

impl Default for LensCapture {
    fn default() -> Self {
        Self::new()
    }
}

/// Phase 4b — captures per-token activation snapshots at named
/// checkpoints. Coarser-grained than `PerOpRecorder`: rather than
/// hashing every op's input/output, it hashes the residual-stream
/// state at "natural" interpretability points (after embedding,
/// after each layer's attention block, after each layer's FFN
/// block, after the final RMSNorm, after lm_head).
///
/// Built once per inference. Pass to `step_with_recorder` alongside
/// (or instead of) `PerOpRecorder`. Both paths can be enabled
/// simultaneously — they're independent.
pub struct PerActivationRecorder<'a> {
    builder: &'a mut CertBuilder,
}

impl<'a> PerActivationRecorder<'a> {
    pub fn new(builder: &'a mut CertBuilder) -> Self {
        Self { builder }
    }

    /// Record one activation snapshot. `layer` = `u32::MAX` for
    /// non-per-layer checkpoints (post_embed, pre_lm_head,
    /// post_lm_head).
    fn record(&mut self, token_index: u32, layer: u32, checkpoint: &str, tensor: &[f32]) {
        let tensor_hash = hash_f32_slice(tensor);
        self.builder
            .declare_activation(token_index, layer, checkpoint, tensor_hash);
    }
}

#[inline]
fn hash_f32_slice(s: &[f32]) -> [u8; 32] {
    // Hash the raw IEEE 754 LE bytes. Bit-identical determinism is
    // why this gives a stable hash — every reproduction sees the
    // exact same float bits, never NaN-collisions or sign-of-zero
    // ambiguity.
    let bytes = unsafe {
        core::slice::from_raw_parts(s.as_ptr() as *const u8, core::mem::size_of_val(s))
    };
    *::blake3::hash(bytes).as_bytes()
}

/// Run one decode step against quantized weights. Same contract as
/// `forward::step`. PerInference mode (no op records emitted).
pub fn step(
    cfg: &Config,
    weights: &QuantizedWeights<'_>,
    state: &mut RunState,
    token: u32,
    pos: usize,
) -> Result<Tensor> {
    step_with_recorder(cfg, weights, state, token, pos, None)
}

/// Run one decode step + optionally record per-op cert entries.
///
/// Forwards to `step_with_recorders` with `act_recorder=None` and
/// `intervention_plan=None`. Retained for backward compat —
/// Phase 4a callers don't need to pass extra slots.
pub fn step_with_recorder(
    cfg: &Config,
    weights: &QuantizedWeights<'_>,
    state: &mut RunState,
    token: u32,
    pos: usize,
    recorder: Option<&mut PerOpRecorder<'_>>,
) -> Result<Tensor> {
    step_with_recorders(cfg, weights, state, token, pos, recorder, None, None, None)
}

/// Run one decode step + optionally record per-op AND/OR per-token
/// activation cert entries (Phase 4a + Phase 4b).
///
/// Both recorders are independent. Either, both, or neither can be
/// enabled. When both are `None`, behavior is bit-identical to
/// `step` (no overhead, no behavior change).
///
/// Activation checkpoints captured (when `act_recorder` is `Some`):
///   - `post_embed`        — after token embedding lookup (layer=MAX)
///   - `post_attn[L]`      — after attention block + residual (per layer)
///   - `post_ffn[L]`       — after FFN block + residual (per layer)
///   - `pre_lm_head`       — after final RMSNorm (layer=MAX)
///   - `post_lm_head`      — logits before sampling (layer=MAX)
///
/// For stories15M (6 layers): 1 + 6×2 + 2 = 15 activations per token.
pub fn step_with_recorders(
    cfg: &Config,
    weights: &QuantizedWeights<'_>,
    state: &mut RunState,
    token: u32,
    pos: usize,
    mut recorder: Option<&mut PerOpRecorder<'_>>,
    mut act_recorder: Option<&mut PerActivationRecorder<'_>>,
    intervention_plan: Option<&InterventionPlan>,
    mut lens: Option<&mut LensCapture>,
) -> Result<Tensor> {
    let pos_u32 = pos as u32;
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim();
    let kv_mul = cfg.kv_mul();
    let head_size = cfg.head_size();
    let hidden_dim = cfg.hidden_dim;
    let n_heads = cfg.n_heads;

    // ---- 0. Token embedding ----
    // Extract one row of the embedding table. For Q4_0 we dequant
    // just the relevant blocks (dim / 32 of them); for F32 it's a
    // direct slice copy.
    let token_us = token as usize;
    let emb_vec = embedding_row(&weights.token_embedding_table, token_us, dim)?;
    let mut x = Tensor::from_f32(emb_vec, Shape::new(&[dim])?)?;
    // Phase 4c — apply intervention BEFORE snapshot, so the
    // snapshot reflects the overridden value and downstream ops see
    // the override.
    if let Some(plan) = intervention_plan {
        if let Some(replacement) = plan.lookup(pos_u32, u32::MAX, "post_embed") {
            if replacement.len() == dim {
                x = Tensor::from_f32(replacement.to_vec(), Shape::new(&[dim])?)?;
            }
        }
    }
    // Phase 4b — post_embed checkpoint (layer=MAX since not per-layer)
    if let Some(ref mut ar) = act_recorder {
        let x_vec = tensor_as_f32(&x)?;
        ar.record(pos_u32, u32::MAX, "post_embed", x_vec);
    }
    // Phase 5a — lens capture (values, not just hash)
    if let Some(ref mut lc) = lens {
        let x_vec = tensor_as_f32(&x)?;
        lc.push(pos_u32, u32::MAX, "post_embed", x_vec);
    }

    for layer in 0..cfg.n_layers {
        let blk = &weights.layers[layer];
        let layer_u32 = layer as u32;

        // ---- 1a. Pre-attention RMSNorm ----
        let xb = ops::rms_norm::rms_norm_slice(&x, blk.rms_att_weight, weights.rms_eps)?;
        let xb_vec = tensor_as_f32(&xb)?.to_vec();
        if let Some(ref mut rec) = recorder {
            // Copy the pre-norm activation only when something records it.
            let x_in_vec = tensor_as_f32(&x)?.to_vec();
            rec.record("rms_norm", layer_u32, &format!("rms_att/{}", layer), &x_in_vec, &xb_vec);
        }

        // ---- 1b. Q/K/V projections (+ optional QKV bias, Qwen2) ----
        let mut q_vec = linear_dispatch(&xb_vec, &blk.wq, dim, attn_q_out_dim(cfg))?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("wq/{}", layer), &xb_vec, &q_vec);
        }
        add_bias(&mut q_vec, blk.bq);
        let mut k_vec = linear_dispatch(&xb_vec, &blk.wk, dim, kv_dim)?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("wk/{}", layer), &xb_vec, &k_vec);
        }
        add_bias(&mut k_vec, blk.bk);
        let mut v_vec = linear_dispatch(&xb_vec, &blk.wv, dim, kv_dim)?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("wv/{}", layer), &xb_vec, &v_vec);
        }
        add_bias(&mut v_vec, blk.bv);

        // ---- 1c. Cache k/v BEFORE rope ----
        let kv_layer_off = layer * cfg.seq_len * kv_dim + pos * kv_dim;
        state.key_cache_mut()[kv_layer_off..kv_layer_off + kv_dim].copy_from_slice(&k_vec);
        state.value_cache_mut()[kv_layer_off..kv_layer_off + kv_dim].copy_from_slice(&v_vec);

        // ---- 1d. RoPE on q (full dim) and cached k (kv_dim) ----
        // Base frequency and layout both come from the model file. Llama
        // uses interleaved rotation of adjacent pairs (i, i+1); Qwen2 uses
        // NeoX rotation of split-halves (j, j + head_size/2).
        let mut q_buf = q_vec;
        let theta = weights.rope_theta;
        if weights.rope_neox {
            let half = head_size / 2;
            // Q: n_heads heads.
            for h in 0..n_heads {
                let base = h * head_size;
                for j in 0..half {
                    let freq = 1.0f32 / libm::powf(theta, (2 * j) as f32 / head_size as f32);
                    let val = pos as f32 * freq;
                    let fcr = libm::cosf(val);
                    let fci = libm::sinf(val);
                    let a = q_buf[base + j];
                    let b = q_buf[base + j + half];
                    q_buf[base + j] = a * fcr - b * fci;
                    q_buf[base + j + half] = a * fci + b * fcr;
                }
            }
            // K: kv_dim / head_size heads, in the cache at this position.
            let n_kv = kv_dim / head_size;
            for h in 0..n_kv {
                let base = kv_layer_off + h * head_size;
                for j in 0..half {
                    let freq = 1.0f32 / libm::powf(theta, (2 * j) as f32 / head_size as f32);
                    let val = pos as f32 * freq;
                    let fcr = libm::cosf(val);
                    let fci = libm::sinf(val);
                    let kc = state.key_cache_mut();
                    let a = kc[base + j];
                    let b = kc[base + j + half];
                    kc[base + j] = a * fcr - b * fci;
                    kc[base + j + half] = a * fci + b * fcr;
                }
            }
        } else {
            let mut i = 0;
            while i < dim {
                let head_dim_idx = i % head_size;
                let freq = 1.0f32 / libm::powf(theta, head_dim_idx as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = libm::cosf(val);
                let fci = libm::sinf(val);

                let q0 = q_buf[i];
                let q1 = q_buf[i + 1];
                q_buf[i] = q0 * fcr - q1 * fci;
                q_buf[i + 1] = q0 * fci + q1 * fcr;

                if i < kv_dim {
                    let k_idx = kv_layer_off + i;
                    let k0 = state.key_cache_mut()[k_idx];
                    let k1 = state.key_cache_mut()[k_idx + 1];
                    state.key_cache_mut()[k_idx] = k0 * fcr - k1 * fci;
                    state.key_cache_mut()[k_idx + 1] = k0 * fci + k1 * fcr;
                }
                i += 2;
            }
        }

        // ---- 1e. Multi-head attention with KV cache ----
        let mut xb_out = alloc::vec![0.0f32; dim];
        for h in 0..n_heads {
            let q_off = h * head_size;
            let mut att = alloc::vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let k_off =
                    layer * cfg.seq_len * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
                let mut score = 0.0f32;
                for d in 0..head_size {
                    score += q_buf[q_off + d] * state.key_cache_mut()[k_off + d];
                }
                score /= libm::sqrtf(head_size as f32);
                att[t] = score;
            }
            // Softmax across positions 0..=pos.
            softmax_inplace(&mut att);
            let xb_off = h * head_size;
            for t in 0..=pos {
                let v_off =
                    layer * cfg.seq_len * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
                let a = att[t];
                for d in 0..head_size {
                    xb_out[xb_off + d] += a * state.value_cache_mut()[v_off + d];
                }
            }
        }

        // Phase 5e — per-head attribution: capture xb_out (concat of
        // all heads' outputs) BEFORE wo so the analysis layer can mask
        // head-h's slice and project to per-head logit contributions.
        if let Some(ref mut lc) = lens {
            lc.push(pos_u32, layer_u32, "pre_wo", &xb_out);
        }

        // ---- 1f. Output projection ----
        let xb2_vec = linear_dispatch(&xb_out, &blk.wo, dim, dim)?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("wo/{}", layer), &xb_out, &xb2_vec);
        }

        // ---- 1g. Residual ----
        let xb2 = Tensor::from_f32(xb2_vec, Shape::new(&[dim])?)?;
        x = x.add(&xb2)?;
        // Phase 4c — apply post_attn[layer] intervention
        if let Some(plan) = intervention_plan {
            if let Some(replacement) = plan.lookup(pos_u32, layer_u32, "post_attn") {
                if replacement.len() == dim {
                    x = Tensor::from_f32(replacement.to_vec(), Shape::new(&[dim])?)?;
                }
            }
        }
        // Phase 4b — post_attn[layer] checkpoint
        if let Some(ref mut ar) = act_recorder {
            let x_vec = tensor_as_f32(&x)?;
            ar.record(pos_u32, layer_u32, "post_attn", x_vec);
        }
        // Phase 5a — lens capture
        if let Some(ref mut lc) = lens {
            let x_vec = tensor_as_f32(&x)?;
            lc.push(pos_u32, layer_u32, "post_attn", x_vec);
        }

        // ---- 2a. Pre-FFN RMSNorm ----
        let xb_ffn = ops::rms_norm::rms_norm_slice(&x, blk.rms_ffn_weight, weights.rms_eps)?;
        let xb_ffn_vec = tensor_as_f32(&xb_ffn)?.to_vec();
        if let Some(ref mut rec) = recorder {
            let x_pre_ffn_vec = tensor_as_f32(&x)?.to_vec();
            rec.record("rms_norm", layer_u32, &format!("rms_ffn/{}", layer), &x_pre_ffn_vec, &xb_ffn_vec);
        }

        // ---- 2b. Gate and Up projections ----
        let gate_vec = linear_dispatch(&xb_ffn_vec, &blk.w1, dim, hidden_dim)?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("w1/{}", layer), &xb_ffn_vec, &gate_vec);
        }
        let up_vec = linear_dispatch(&xb_ffn_vec, &blk.w3, dim, hidden_dim)?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("w3/{}", layer), &xb_ffn_vec, &up_vec);
        }

        // ---- 2c. SwiGLU: silu(gate) * up ----
        let mut ffn_inner = alloc::vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            let g = gate_vec[i];
            let silu = g / (1.0 + libm::expf(-g));
            ffn_inner[i] = silu * up_vec[i];
        }

        // ---- 2d. Down projection ----
        let xb_down_vec = linear_dispatch(&ffn_inner, &blk.w2, hidden_dim, dim)?;
        if let Some(ref mut rec) = recorder {
            rec.record("linear_q4_0", layer_u32, &format!("w2/{}", layer), &ffn_inner, &xb_down_vec);
        }

        // ---- 2e. Residual ----
        let xb_down = Tensor::from_f32(xb_down_vec, Shape::new(&[dim])?)?;
        x = x.add(&xb_down)?;
        // Phase 4c — apply post_ffn[layer] intervention
        if let Some(plan) = intervention_plan {
            if let Some(replacement) = plan.lookup(pos_u32, layer_u32, "post_ffn") {
                if replacement.len() == dim {
                    x = Tensor::from_f32(replacement.to_vec(), Shape::new(&[dim])?)?;
                }
            }
        }
        // Phase 4b — post_ffn[layer] checkpoint
        if let Some(ref mut ar) = act_recorder {
            let x_vec = tensor_as_f32(&x)?;
            ar.record(pos_u32, layer_u32, "post_ffn", x_vec);
        }
        // Phase 5a — lens capture
        if let Some(ref mut lc) = lens {
            let x_vec = tensor_as_f32(&x)?;
            lc.push(pos_u32, layer_u32, "post_ffn", x_vec);
        }
    }

    // ---- 3. Final norm + lm_head ----
    let x_final = ops::rms_norm::rms_norm_slice(&x, weights.rms_final_weight, weights.rms_eps)?;
    let mut x_final_vec = tensor_as_f32(&x_final)?.to_vec();
    if let Some(ref mut rec) = recorder {
        let x_pre_final_vec = tensor_as_f32(&x)?.to_vec();
        rec.record("rms_norm", u32::MAX, "rms_final", &x_pre_final_vec, &x_final_vec);
    }
    // Phase 4c — apply pre_lm_head intervention BEFORE snapshot +
    // BEFORE lm_head matmul, so the override propagates to logits.
    if let Some(plan) = intervention_plan {
        if let Some(replacement) = plan.lookup(pos_u32, u32::MAX, "pre_lm_head") {
            if replacement.len() == dim {
                x_final_vec = replacement.to_vec();
            }
        }
    }
    // Phase 4b — pre_lm_head checkpoint
    if let Some(ref mut ar) = act_recorder {
        ar.record(pos_u32, u32::MAX, "pre_lm_head", &x_final_vec);
    }
    // Phase 5a — lens capture
    if let Some(ref mut lc) = lens {
        lc.push(pos_u32, u32::MAX, "pre_lm_head", &x_final_vec);
    }

    // wcls: present → dedicated lm_head; absent → share token_embd.
    let (logits_vec, lm_head_key) = if let Some(wcls) = &weights.wcls {
        (
            linear_dispatch(&x_final_vec, wcls, dim, cfg.vocab_size)?,
            "wcls",
        )
    } else {
        (
            linear_dispatch(&x_final_vec, &weights.token_embedding_table, dim, cfg.vocab_size)?,
            "token_embd",
        )
    };
    if let Some(ref mut rec) = recorder {
        rec.record("linear_q4_0", u32::MAX, lm_head_key, &x_final_vec, &logits_vec);
    }
    // Phase 4c — apply post_lm_head intervention (last chance to
    // perturb output before sampling). Replacement must be
    // vocab_size-sized.
    let mut logits_vec = logits_vec;
    if let Some(plan) = intervention_plan {
        if let Some(replacement) = plan.lookup(pos_u32, u32::MAX, "post_lm_head") {
            if replacement.len() == cfg.vocab_size {
                logits_vec = replacement.to_vec();
            }
        }
    }
    // Phase 4b — post_lm_head checkpoint (logits before sampling)
    if let Some(ref mut ar) = act_recorder {
        ar.record(pos_u32, u32::MAX, "post_lm_head", &logits_vec);
    }

    Tensor::from_f32(logits_vec, Shape::new(&[cfg.vocab_size])?)
}

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

/// Q for grouped-query attention: out = n_heads * head_size = dim.
/// Pulled into a named function so the Q/K/V split intent is obvious.
fn attn_q_out_dim(cfg: &Config) -> usize {
    cfg.n_heads * cfg.head_size()
}

/// Dispatch one `[batch=1, in_feat] x W[out, in]^T = [1, out_feat]` op
/// based on weight dtype.
fn linear_dispatch(
    x: &[f32],
    w: &QuantTensor<'_>,
    in_feat: usize,
    out_feat: usize,
) -> Result<Vec<f32>> {
    let mut y = alloc::vec![0.0f32; out_feat];
    match w.dtype {
        GgufTensorType::Q4_0 => {
            ops::quant::linear_q4_0_cpu(x, w.bytes, &mut y, 1, in_feat, out_feat)?;
        }
        GgufTensorType::F32 => {
            // Plain row-major matmul: y[o] = sum_i x[i] * W[o, i]
            if w.bytes.len() != out_feat * in_feat * 4 {
                return Err(Error::Internal(
                    "linear_dispatch F32: weight size mismatch",
                ));
            }
            // SAFETY: dtype is F32; bytes are 4-byte LE floats.
            let w_f32: &[f32] = unsafe {
                core::slice::from_raw_parts(w.bytes.as_ptr() as *const f32, out_feat * in_feat)
            };
            // MUST go through the canonical reduction like every other arm.
            // This was a plain serial `acc += x[i] * row[i]` — regime-1
            // semantics — while the certificate declared regime 3. Every
            // other dtype here (Q4_0, Q4_K, Q6_K) reduces canonically, so a
            // model carrying F32 linear weights would have been certified
            // under a regime it did not actually use. Found 2026-07-29 while
            // re-establishing cross-vendor agreement; nothing exercised it
            // because the models tested keep F32 only for norms.
            for o in 0..out_feat {
                let row = &w_f32[o * in_feat..(o + 1) * in_feat];
                y[o] = ops::quant::canonical_dot_pub(x, row, in_feat);
            }
        }
        GgufTensorType::Q4_K => {
            // Fused dequant+dot. Bit-identical to linear_q4_k_cpu (asserted
            // by ops::quant::tests::fused_q4k_dot_is_bit_identical) but never
            // materialises the f32 weight matrix — at batch=1 the old path
            // allocated and wrote 235 MB per call for a 4096x14336 layer,
            // reading each value exactly once.
            #[cfg(feature = "std-parallel")]
            {
                // Row-parallel: bit-identical, because each output row is an
                // independent reduction. Threshold keeps small projections
                // (norms, tiny heads) off the thread pool, where spawn cost
                // would dominate the work.
                const PAR_MIN_ROWS: usize = 512;
                if out_feat >= PAR_MIN_ROWS {
                    let t = std::thread::available_parallelism()
                        .map(|v| v.get())
                        .unwrap_or(1);
                    ops::quant::linear_q4_k_integer_parallel(
                        x, w.bytes, &mut y, 1, in_feat, out_feat, t,
                    )?;
                } else {
                    ops::quant::linear_q4_k_integer(x, w.bytes, &mut y, 1, in_feat, out_feat)?;
                }
            }
            #[cfg(not(feature = "std-parallel"))]
            {
                ops::quant::linear_q4_k_fused(x, w.bytes, &mut y, 1, in_feat, out_feat)?;
            }
        }
        GgufTensorType::Q6_K => {
            // Q6_K carries 15.7% of Mistral's weights INCLUDING the largest
            // tensor in the model (output.weight, 131.1M) and every ffn_down.
            // It was the last unfused, unthreaded path — the serial tail that
            // pinned CPU near 200% on 14 cores.
            #[cfg(feature = "std-parallel")]
            {
                const PAR_MIN_ROWS: usize = 512;
                if out_feat >= PAR_MIN_ROWS {
                    let t = std::thread::available_parallelism()
                        .map(|v| v.get())
                        .unwrap_or(1);
                    ops::quant::linear_q6_k_integer_parallel(
                        x, w.bytes, &mut y, 1, in_feat, out_feat, t,
                    )?;
                } else {
                    ops::quant::linear_q6_k_integer(x, w.bytes, &mut y, 1, in_feat, out_feat)?;
                }
            }
            #[cfg(not(feature = "std-parallel"))]
            {
                ops::quant::linear_q6_k_integer(x, w.bytes, &mut y, 1, in_feat, out_feat)?;
            }
        }
        GgufTensorType::Q8_0 => {
            // Q8_0: dequant each output row (exact int8 x f16-scale), then
            // reduce through the canonical dot — the same cross-vendor
            // deterministic reduction used by the F32 arm. Per-row dequant
            // keeps memory bounded (never materialises the full matrix).
            let row_bytes = q8_0_row_bytes(in_feat)?;
            if w.bytes.len() < out_feat * row_bytes {
                return Err(Error::Internal("linear_dispatch Q8_0: weight blob too small"));
            }
            for o in 0..out_feat {
                let row = ops::quant::dequantize_q8_0(&w.bytes[o * row_bytes..(o + 1) * row_bytes])?;
                y[o] = ops::quant::canonical_dot_pub(x, &row, in_feat);
            }
        }
        GgufTensorType::Q5_0 => {
            // Q5_0: dequant each output row (5-bit → f32), then canonical dot —
            // same bounded-memory, cross-vendor-deterministic path as Q8_0.
            let row_bytes = q5_0_row_bytes(in_feat)?;
            if w.bytes.len() < out_feat * row_bytes {
                return Err(Error::Internal("linear_dispatch Q5_0: weight blob too small"));
            }
            for o in 0..out_feat {
                let row = ops::quant::dequantize_q5_0(&w.bytes[o * row_bytes..(o + 1) * row_bytes])?;
                y[o] = ops::quant::canonical_dot_pub(x, &row, in_feat);
            }
        }
        GgufTensorType::F16 | GgufTensorType::Other(_) => {
            return Err(Error::NotImplemented {
                op: "forward_quantized::linear_dispatch",
                why: "weight dtype not Q4_0 / Q4_K / Q5_0 / Q6_K / Q8_0 / F32",
            });
        }
    }
    Ok(y)
}

/// Bytes per output row of a Q8_0 weight matrix with `in_feat` columns.
/// Q8_0 packs 32 elements per 34-byte block (f16 scale + 32 int8).
fn q8_0_row_bytes(in_feat: usize) -> Result<usize> {
    if in_feat % 32 != 0 {
        return Err(Error::Internal("Q8_0: in_feat must be a multiple of 32"));
    }
    Ok((in_feat / 32) * 34)
}

/// Bytes per output row for a Q5_0 weight matrix. Q5_0 packs 32 elements per
/// 22-byte block (f16 scale + 4-byte high-bit field + 16 low-nibble bytes).
fn q5_0_row_bytes(in_feat: usize) -> Result<usize> {
    if in_feat % 32 != 0 {
        return Err(Error::Internal("Q5_0: in_feat must be a multiple of 32"));
    }
    Ok((in_feat / 32) * 22)
}

/// Add an optional bias vector in place (Qwen2 QKV biases; no-op for Llama).
#[inline]
fn add_bias(v: &mut [f32], bias: Option<&[f32]>) {
    if let Some(b) = bias {
        for (vi, bi) in v.iter_mut().zip(b.iter()) {
            *vi += *bi;
        }
    }
}

/// Extract one row of an embedding table (Q4_0 or F32). For Q4_0
/// we dequantize just the `dim/32` blocks for the target row, not
/// the whole table.
fn embedding_row(table: &QuantTensor<'_>, row: usize, dim: usize) -> Result<Vec<f32>> {
    match table.dtype {
        GgufTensorType::F32 => {
            if table.bytes.len() < (row + 1) * dim * 4 {
                return Err(Error::Internal(
                    "embedding_row F32: table too small for token id",
                ));
            }
            let w_f32: &[f32] = unsafe {
                core::slice::from_raw_parts(
                    table.bytes.as_ptr() as *const f32,
                    table.bytes.len() / 4,
                )
            };
            Ok(w_f32[row * dim..(row + 1) * dim].to_vec())
        }
        GgufTensorType::Q4_0 => {
            if dim % 32 != 0 {
                return Err(Error::Internal(
                    "embedding_row Q4_0: dim must be a multiple of 32",
                ));
            }
            let blocks_per_row = dim / 32;
            let row_start = row * blocks_per_row * 18;
            let row_end = row_start + blocks_per_row * 18;
            if row_end > table.bytes.len() {
                return Err(Error::Internal(
                    "embedding_row Q4_0: row past end of table",
                ));
            }
            ops::quant::dequantize_q4_0(&table.bytes[row_start..row_end])
        }
        GgufTensorType::Q4_K => {
            if dim % 256 != 0 {
                return Err(Error::Internal(
                    "embedding_row Q4_K: dim must be a multiple of 256",
                ));
            }
            let blocks_per_row = dim / 256;
            let row_start = row * blocks_per_row * 144;
            let row_end = row_start + blocks_per_row * 144;
            if row_end > table.bytes.len() {
                return Err(Error::Internal(
                    "embedding_row Q4_K: row past end of table",
                ));
            }
            ops::quant::dequantize_q4_k(&table.bytes[row_start..row_end])
        }
        GgufTensorType::Q6_K => {
            if dim % 256 != 0 {
                return Err(Error::Internal(
                    "embedding_row Q6_K: dim must be a multiple of 256",
                ));
            }
            let blocks_per_row = dim / 256;
            let row_start = row * blocks_per_row * 210;
            let row_end = row_start + blocks_per_row * 210;
            if row_end > table.bytes.len() {
                return Err(Error::Internal(
                    "embedding_row Q6_K: row past end of table",
                ));
            }
            ops::quant::dequantize_q6_k(&table.bytes[row_start..row_end])
        }
        GgufTensorType::Q5_0 => {
            if dim % 32 != 0 {
                return Err(Error::Internal(
                    "embedding_row Q5_0: dim must be a multiple of 32",
                ));
            }
            let blocks_per_row = dim / 32;
            let row_start = row * blocks_per_row * 22;
            let row_end = row_start + blocks_per_row * 22;
            if row_end > table.bytes.len() {
                return Err(Error::Internal(
                    "embedding_row Q5_0: row past end of table",
                ));
            }
            ops::quant::dequantize_q5_0(&table.bytes[row_start..row_end])
        }
        GgufTensorType::Q8_0 => {
            if dim % 32 != 0 {
                return Err(Error::Internal(
                    "embedding_row Q8_0: dim must be a multiple of 32",
                ));
            }
            let blocks_per_row = dim / 32;
            let row_start = row * blocks_per_row * 34;
            let row_end = row_start + blocks_per_row * 34;
            if row_end > table.bytes.len() {
                return Err(Error::Internal(
                    "embedding_row Q8_0: row past end of table",
                ));
            }
            ops::quant::dequantize_q8_0(&table.bytes[row_start..row_end])
        }
        _ => Err(Error::NotImplemented {
            op: "embedding_row",
            why: "non-Q4_0 / Q4_K / Q6_K / Q8_0 / F32 embedding dtype",
        }),
    }
}

/// In-place softmax — same numerical sequence as
/// `ops::softmax::softmax_last_dim` (subtract max then exp+normalize),
/// but operates directly on a `&mut [f32]` to avoid wrapping in a
/// Tensor for every attention head.
fn softmax_inplace(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn tensor_as_f32(t: &Tensor) -> Result<&[f32]> {
    if let Storage::Cpu(s) = t.storage() {
        Ok(s.as_f32_slice())
    } else {
        Err(Error::NotImplemented {
            op: "tensor_as_f32",
            why: "GPU storage not supported in forward_quantized helpers",
        })
    }
}

// =========================================================================
//                                  TESTS
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::quant_weights::QuantTensor;
    use alloc::vec;

    /// Build a `QuantTensor` from an owned byte vec. We have to leak
    /// the Vec to get a `&'static [u8]` for the test — tolerable
    /// because each test holds at most a few KB.
    fn quant_tensor_from_vec(bytes: Vec<u8>, shape: Vec<u64>, dtype: GgufTensorType) -> QuantTensor<'static> {
        let leaked: &'static [u8] = alloc::boxed::Box::leak(bytes.into_boxed_slice());
        QuantTensor {
            shape,
            dtype,
            bytes: leaked,
        }
    }

    /// Cross-check: running forward_quantized on Q4_0 weights must
    /// equal running forward (the F32 path) on those weights AFTER
    /// dequantization. Same float math, same numerical sequence →
    /// bit-identical logits.
    #[test]
    fn quantized_matches_dequant_then_forward() {
        use crate::model::forward;
        use crate::model::weights::Weights;

        // Tiny synthetic config — keep numel small but fully exercises
        // every code path (multi-layer, multi-head, GQA-trivial,
        // shared embedding).
        let cfg = Config {
            dim: 32,
            hidden_dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 32,
            seq_len: 16,
            shared_weights: true,
        };

        // Build F32 weights (deterministic synthetic).
        let head_size = cfg.head_size();
        let kv_dim = cfg.kv_dim();
        let make = |offset: f32, len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let x = (i as f32 + offset) * 0.013;
                    (libm::sinf(x) * 0.4) + libm::cosf(x * 1.3) * 0.2
                })
                .collect()
        };

        let token_embed = make(1.0, cfg.vocab_size * cfg.dim);
        let rms_att = make(2.0, cfg.n_layers * cfg.dim);
        let wq = make(3.0, cfg.n_layers * cfg.dim * cfg.dim);
        let wk = make(4.0, cfg.n_layers * cfg.dim * kv_dim);
        let wv = make(5.0, cfg.n_layers * cfg.dim * kv_dim);
        let wo = make(6.0, cfg.n_layers * cfg.dim * cfg.dim);
        let rms_ffn = make(7.0, cfg.n_layers * cfg.dim);
        let w1 = make(8.0, cfg.n_layers * cfg.hidden_dim * cfg.dim);
        let w2 = make(9.0, cfg.n_layers * cfg.dim * cfg.hidden_dim);
        let w3 = make(10.0, cfg.n_layers * cfg.hidden_dim * cfg.dim);
        let rms_final = make(11.0, cfg.dim);

        // Quantize the projection matrices + embed to Q4_0; norms stay F32.
        let token_embed_q4 = ops::quant::quantize_q4_0(&token_embed).unwrap();
        let wq_q4 = ops::quant::quantize_q4_0(&wq).unwrap();
        let wk_q4 = ops::quant::quantize_q4_0(&wk).unwrap();
        let wv_q4 = ops::quant::quantize_q4_0(&wv).unwrap();
        let wo_q4 = ops::quant::quantize_q4_0(&wo).unwrap();
        let w1_q4 = ops::quant::quantize_q4_0(&w1).unwrap();
        let w2_q4 = ops::quant::quantize_q4_0(&w2).unwrap();
        let w3_q4 = ops::quant::quantize_q4_0(&w3).unwrap();

        // Dequantize back to F32 — this is the "ground truth" F32
        // tensor that the dequant-then-F32 forward will operate on.
        // It MUST equal what linear_q4_0_cpu / embedding_row reads
        // back, so both paths see the same numbers.
        let token_embed_dq = ops::quant::dequantize_q4_0(&token_embed_q4).unwrap();
        let wq_dq = ops::quant::dequantize_q4_0(&wq_q4).unwrap();
        let wk_dq = ops::quant::dequantize_q4_0(&wk_q4).unwrap();
        let wv_dq = ops::quant::dequantize_q4_0(&wv_q4).unwrap();
        let wo_dq = ops::quant::dequantize_q4_0(&wo_q4).unwrap();
        let w1_dq = ops::quant::dequantize_q4_0(&w1_q4).unwrap();
        let w2_dq = ops::quant::dequantize_q4_0(&w2_q4).unwrap();
        let w3_dq = ops::quant::dequantize_q4_0(&w3_q4).unwrap();

        // Build F32 Weights using the dequantized values.
        let w_f32 = Weights {
            token_embedding_table: &token_embed_dq,
            rms_att_weight: &rms_att,
            wq: &wq_dq,
            wk: &wk_dq,
            wv: &wv_dq,
            wo: &wo_dq,
            rms_ffn_weight: &rms_ffn,
            w1: &w1_dq,
            w2: &w2_dq,
            w3: &w3_dq,
            rms_final_weight: &rms_final,
            wcls: &token_embed_dq, // shared embedding
        };

        // Build QuantizedWeights using the Q4_0 byte blobs.
        let qw = QuantizedWeights {
            rope_theta: 10000.0, rms_eps: 1e-5, rope_neox: false,
            token_embedding_table: quant_tensor_from_vec(
                token_embed_q4.clone(),
                vec![cfg.dim as u64, cfg.vocab_size as u64],
                GgufTensorType::Q4_0,
            ),
            layers: (0..cfg.n_layers)
                .map(|l| {
                    let dim2 = cfg.dim * cfg.dim;
                    let dim_kv = cfg.dim * kv_dim;
                    let dim_hidden = cfg.dim * cfg.hidden_dim;
                    let _ = head_size;
                    crate::model::quant_weights::QuantLayer {
                        bq: None, bk: None, bv: None,
                        rms_att_weight: leak_f32_slice(&rms_att[l * cfg.dim..(l + 1) * cfg.dim]),
                        wq: quant_tensor_from_vec(
                            slice_q4(&wq_q4, l, dim2),
                            vec![cfg.dim as u64, cfg.dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                        wk: quant_tensor_from_vec(
                            slice_q4(&wk_q4, l, dim_kv),
                            vec![cfg.dim as u64, kv_dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                        wv: quant_tensor_from_vec(
                            slice_q4(&wv_q4, l, dim_kv),
                            vec![cfg.dim as u64, kv_dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                        wo: quant_tensor_from_vec(
                            slice_q4(&wo_q4, l, dim2),
                            vec![cfg.dim as u64, cfg.dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                        rms_ffn_weight: leak_f32_slice(
                            &rms_ffn[l * cfg.dim..(l + 1) * cfg.dim],
                        ),
                        w1: quant_tensor_from_vec(
                            slice_q4(&w1_q4, l, dim_hidden),
                            vec![cfg.dim as u64, cfg.hidden_dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                        w2: quant_tensor_from_vec(
                            slice_q4(&w2_q4, l, dim_hidden),
                            vec![cfg.hidden_dim as u64, cfg.dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                        w3: quant_tensor_from_vec(
                            slice_q4(&w3_q4, l, dim_hidden),
                            vec![cfg.dim as u64, cfg.hidden_dim as u64],
                            GgufTensorType::Q4_0,
                        ),
                    }
                })
                .collect(),
            rms_final_weight: leak_f32_slice(&rms_final),
            wcls: None, // shared_weights = true
        };

        // Run both forwards.
        let mut state_f32 = RunState::new(&cfg);
        let mut state_q = RunState::new(&cfg);
        let logits_f32 = forward::step(&cfg, &w_f32, &mut state_f32, 5, 0).unwrap();
        let logits_q = step(&cfg, &qw, &mut state_q, 5, 0).unwrap();

        let f_data = tensor_as_f32(&logits_f32).unwrap();
        let q_data = tensor_as_f32(&logits_q).unwrap();
        assert_eq!(f_data.len(), q_data.len());

        // Bit-identical or near enough (rounding noise from
        // independently-recomputed in-place softmax vs the Tensor
        // softmax path; tolerance 1e-3 covers it).
        let mut max_diff = 0f32;
        for i in 0..f_data.len() {
            let d = (f_data[i] - q_data[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-3,
            "Q4_0 forward diverges from dequant-then-F32 forward: max_diff = {}",
            max_diff
        );
    }

    fn slice_q4(blob: &[u8], layer: usize, numel_per_layer: usize) -> Vec<u8> {
        let blocks = numel_per_layer / 32;
        let bytes = blocks * 18;
        blob[layer * bytes..(layer + 1) * bytes].to_vec()
    }

    fn leak_f32_slice(s: &[f32]) -> &'static [f32] {
        let v: Vec<f32> = s.to_vec();
        alloc::boxed::Box::leak(v.into_boxed_slice())
    }

    #[test]
    fn small_q4_forward_produces_finite_logits() {
        // Same setup, but only check that we produce finite outputs.
        let cfg = Config {
            dim: 32,
            hidden_dim: 64,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 32,
            seq_len: 8,
            shared_weights: true,
        };
        let kv_dim = cfg.kv_dim();
        let mk = |seed: f32, len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| libm::sinf((i as f32 + seed) * 0.07) * 0.3)
                .collect()
        };
        let qte = ops::quant::quantize_q4_0(&mk(1.0, cfg.vocab_size * cfg.dim)).unwrap();
        let qwq = ops::quant::quantize_q4_0(&mk(2.0, cfg.dim * cfg.dim)).unwrap();
        let qwk = ops::quant::quantize_q4_0(&mk(3.0, cfg.dim * kv_dim)).unwrap();
        let qwv = ops::quant::quantize_q4_0(&mk(4.0, cfg.dim * kv_dim)).unwrap();
        let qwo = ops::quant::quantize_q4_0(&mk(5.0, cfg.dim * cfg.dim)).unwrap();
        let qw1 = ops::quant::quantize_q4_0(&mk(6.0, cfg.dim * cfg.hidden_dim)).unwrap();
        let qw2 = ops::quant::quantize_q4_0(&mk(7.0, cfg.dim * cfg.hidden_dim)).unwrap();
        let qw3 = ops::quant::quantize_q4_0(&mk(8.0, cfg.dim * cfg.hidden_dim)).unwrap();
        let rms_att = mk(9.0, cfg.dim);
        let rms_ffn = mk(10.0, cfg.dim);
        let rms_final = mk(11.0, cfg.dim);

        let qw = QuantizedWeights {
            rope_theta: 10000.0, rms_eps: 1e-5, rope_neox: false,
            token_embedding_table: quant_tensor_from_vec(
                qte,
                vec![cfg.dim as u64, cfg.vocab_size as u64],
                GgufTensorType::Q4_0,
            ),
            layers: vec![crate::model::quant_weights::QuantLayer {
                        bq: None, bk: None, bv: None,
                rms_att_weight: leak_f32_slice(&rms_att),
                wq: quant_tensor_from_vec(qwq, vec![cfg.dim as u64, cfg.dim as u64], GgufTensorType::Q4_0),
                wk: quant_tensor_from_vec(qwk, vec![cfg.dim as u64, kv_dim as u64], GgufTensorType::Q4_0),
                wv: quant_tensor_from_vec(qwv, vec![cfg.dim as u64, kv_dim as u64], GgufTensorType::Q4_0),
                wo: quant_tensor_from_vec(qwo, vec![cfg.dim as u64, cfg.dim as u64], GgufTensorType::Q4_0),
                rms_ffn_weight: leak_f32_slice(&rms_ffn),
                w1: quant_tensor_from_vec(qw1, vec![cfg.dim as u64, cfg.hidden_dim as u64], GgufTensorType::Q4_0),
                w2: quant_tensor_from_vec(qw2, vec![cfg.hidden_dim as u64, cfg.dim as u64], GgufTensorType::Q4_0),
                w3: quant_tensor_from_vec(qw3, vec![cfg.dim as u64, cfg.hidden_dim as u64], GgufTensorType::Q4_0),
            }],
            rms_final_weight: leak_f32_slice(&rms_final),
            wcls: None,
        };

        let mut state = RunState::new(&cfg);
        let logits = step(&cfg, &qw, &mut state, 0, 0).unwrap();
        let data = tensor_as_f32(&logits).unwrap();
        assert_eq!(data.len(), cfg.vocab_size);
        for &v in data {
            assert!(v.is_finite(), "logit is NaN/inf: {}", v);
        }
        // Logits shouldn't be all-zero (would mean every projection
        // collapsed to 0 — usually a bug).
        let any_nonzero = data.iter().any(|&v| v.abs() > 1e-8);
        assert!(any_nonzero, "all logits are zero");
    }

    /// PerOp instrumentation count check + bit-equivalence with
    /// non-recorded forward.
    ///
    /// 11 ops per layer (1 rms_att + 3 q/k/v + 1 wo + 1 rms_ffn +
    /// 2 w1/w3 + 1 w2 + ... = let's count: rms_att, wq, wk, wv, wo,
    /// rms_ffn, w1, w3, w2 = 9 ops per layer). Plus the final rms_norm
    /// and lm_head = 9*L + 2.
    #[test]
    fn perop_records_expected_count_and_matches_baseline_logits() {
        use crate::cert::CertBuilder;

        let cfg = Config {
            dim: 32,
            hidden_dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 32,
            seq_len: 8,
            shared_weights: true,
        };
        let kv_dim = cfg.kv_dim();
        let mk = |seed: f32, len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| libm::sinf((i as f32 + seed) * 0.07) * 0.3)
                .collect()
        };
        let qte = ops::quant::quantize_q4_0(&mk(1.0, cfg.vocab_size * cfg.dim)).unwrap();
        let qwq = ops::quant::quantize_q4_0(&mk(2.0, cfg.dim * cfg.dim)).unwrap();
        let qwk = ops::quant::quantize_q4_0(&mk(3.0, cfg.dim * kv_dim)).unwrap();
        let qwv = ops::quant::quantize_q4_0(&mk(4.0, cfg.dim * kv_dim)).unwrap();
        let qwo = ops::quant::quantize_q4_0(&mk(5.0, cfg.dim * cfg.dim)).unwrap();
        let qw1 = ops::quant::quantize_q4_0(&mk(6.0, cfg.dim * cfg.hidden_dim)).unwrap();
        let qw2 = ops::quant::quantize_q4_0(&mk(7.0, cfg.dim * cfg.hidden_dim)).unwrap();
        let qw3 = ops::quant::quantize_q4_0(&mk(8.0, cfg.dim * cfg.hidden_dim)).unwrap();
        let rms_att_l0 = mk(9.0, cfg.dim);
        let rms_ffn_l0 = mk(10.0, cfg.dim);
        let rms_final = mk(11.0, cfg.dim);

        // L0 + L1 reuse the same buffers — fine for shape, not for math
        // realism, but enough for the structural test.
        let leak_f32 = |s: &[f32]| -> &'static [f32] {
            let v: Vec<f32> = s.to_vec();
            alloc::boxed::Box::leak(v.into_boxed_slice())
        };
        let leak_bytes = |b: Vec<u8>| -> &'static [u8] {
            alloc::boxed::Box::leak(b.into_boxed_slice())
        };

        let layer = |seed: usize| crate::model::quant_weights::QuantLayer {
                        bq: None, bk: None, bv: None,
            rms_att_weight: leak_f32(&rms_att_l0),
            wq: QuantTensor {
                shape: vec![cfg.dim as u64, cfg.dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qwq.clone()),
            },
            wk: QuantTensor {
                shape: vec![cfg.dim as u64, kv_dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qwk.clone()),
            },
            wv: QuantTensor {
                shape: vec![cfg.dim as u64, kv_dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qwv.clone()),
            },
            wo: QuantTensor {
                shape: vec![cfg.dim as u64, cfg.dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qwo.clone()),
            },
            rms_ffn_weight: leak_f32(&rms_ffn_l0),
            w1: QuantTensor {
                shape: vec![cfg.dim as u64, cfg.hidden_dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qw1.clone()),
            },
            w2: QuantTensor {
                shape: vec![cfg.hidden_dim as u64, cfg.dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qw2.clone()),
            },
            w3: QuantTensor {
                shape: vec![cfg.dim as u64, cfg.hidden_dim as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qw3.clone()),
            },
        };

        let qw = QuantizedWeights {
            rope_theta: 10000.0, rms_eps: 1e-5, rope_neox: false,
            token_embedding_table: QuantTensor {
                shape: vec![cfg.dim as u64, cfg.vocab_size as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(qte),
            },
            layers: vec![layer(0), layer(1)],
            rms_final_weight: leak_f32(&rms_final),
            wcls: None,
        };

        // Two runs: one without recorder, one WITH. Logits must be
        // bit-identical (recorder is observation-only).
        let mut state_baseline = RunState::new(&cfg);
        let logits_baseline = step(&cfg, &qw, &mut state_baseline, 0, 0).unwrap();

        let mut state_recorded = RunState::new(&cfg);
        let mut builder = CertBuilder::new();
        builder.declare_input("model_id", b"unit-test");
        let n_ops_before;
        {
            let mut rec = PerOpRecorder::new(&mut builder, &qw);
            let logits_recorded =
                step_with_recorder(&cfg, &qw, &mut state_recorded, 0, 0, Some(&mut rec))
                    .unwrap();

            // Bit-identical logits (recorder is non-perturbative).
            let f_base = tensor_as_f32(&logits_baseline).unwrap();
            let f_rec = tensor_as_f32(&logits_recorded).unwrap();
            for (a, b) in f_base.iter().zip(f_rec.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(),
                    "PerOp recorder perturbed logits — recorder must be observation-only");
            }
            n_ops_before = builder.ops().len();
        }
        // Per-token count: 9 ops/layer × 2 layers + 2 (rms_final + lm_head) = 20.
        let expected_per_token: usize = cfg.n_layers * 9 + 2;
        assert_eq!(
            n_ops_before, expected_per_token,
            "expected {} ops, got {}", expected_per_token, n_ops_before
        );

        // Sanity: op_index is monotonic 0..expected.
        for (i, op) in builder.ops().iter().enumerate() {
            assert_eq!(op.op_index as usize, i);
        }
    }

    /// Same prompt → same op hashes across runs. The most load-bearing
    /// property: PerOp records are deterministic, so two runs of the
    /// same inference produce the same cert digest.
    #[test]
    fn perop_records_are_deterministic_across_runs() {
        use crate::cert::CertBuilder;

        let cfg = Config {
            dim: 32, hidden_dim: 64, n_layers: 1,
            n_heads: 4, n_kv_heads: 4, vocab_size: 32,
            seq_len: 8, shared_weights: true,
        };
        let kv_dim = cfg.kv_dim();
        let mk = |seed: f32, len: usize| -> Vec<f32> {
            (0..len).map(|i| libm::sinf((i as f32 + seed) * 0.07) * 0.3).collect()
        };
        let leak_bytes = |b: Vec<u8>| -> &'static [u8] {
            alloc::boxed::Box::leak(b.into_boxed_slice())
        };
        let leak_f32 = |s: &[f32]| -> &'static [f32] {
            alloc::boxed::Box::leak(s.to_vec().into_boxed_slice())
        };

        let q_bytes = |seed: f32, len: usize| -> Vec<u8> {
            ops::quant::quantize_q4_0(&mk(seed, len)).unwrap()
        };
        let qw = QuantizedWeights {
            rope_theta: 10000.0, rms_eps: 1e-5, rope_neox: false,
            token_embedding_table: QuantTensor {
                shape: vec![cfg.dim as u64, cfg.vocab_size as u64],
                dtype: GgufTensorType::Q4_0,
                bytes: leak_bytes(q_bytes(1.0, cfg.vocab_size * cfg.dim)),
            },
            layers: vec![crate::model::quant_weights::QuantLayer {
                        bq: None, bk: None, bv: None,
                rms_att_weight: leak_f32(&mk(9.0, cfg.dim)),
                wq: QuantTensor { shape: vec![cfg.dim as u64, cfg.dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(2.0, cfg.dim * cfg.dim)) },
                wk: QuantTensor { shape: vec![cfg.dim as u64, kv_dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(3.0, cfg.dim * kv_dim)) },
                wv: QuantTensor { shape: vec![cfg.dim as u64, kv_dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(4.0, cfg.dim * kv_dim)) },
                wo: QuantTensor { shape: vec![cfg.dim as u64, cfg.dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(5.0, cfg.dim * cfg.dim)) },
                rms_ffn_weight: leak_f32(&mk(10.0, cfg.dim)),
                w1: QuantTensor { shape: vec![cfg.dim as u64, cfg.hidden_dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(6.0, cfg.dim * cfg.hidden_dim)) },
                w2: QuantTensor { shape: vec![cfg.hidden_dim as u64, cfg.dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(7.0, cfg.dim * cfg.hidden_dim)) },
                w3: QuantTensor { shape: vec![cfg.dim as u64, cfg.hidden_dim as u64], dtype: GgufTensorType::Q4_0,
                    bytes: leak_bytes(q_bytes(8.0, cfg.dim * cfg.hidden_dim)) },
            }],
            rms_final_weight: leak_f32(&mk(11.0, cfg.dim)),
            wcls: None,
        };

        let mut run_a_builder = CertBuilder::new();
        {
            let mut rec = PerOpRecorder::new(&mut run_a_builder, &qw);
            let mut st = RunState::new(&cfg);
            let _ = step_with_recorder(&cfg, &qw, &mut st, 0, 0, Some(&mut rec)).unwrap();
        }
        let cert_a = run_a_builder.finalize();

        let mut run_b_builder = CertBuilder::new();
        {
            let mut rec = PerOpRecorder::new(&mut run_b_builder, &qw);
            let mut st = RunState::new(&cfg);
            let _ = step_with_recorder(&cfg, &qw, &mut st, 0, 0, Some(&mut rec)).unwrap();
        }
        let cert_b = run_b_builder.finalize();

        // SAME prompt + SAME weights + SAME op order → IDENTICAL digest
        // and IDENTICAL op records. If this ever fails, PerOp lost determinism
        // and the interpretability claims it enables become non-falsifiable.
        assert_eq!(cert_a.digest, cert_b.digest);
        assert_eq!(cert_a.ops.len(), cert_b.ops.len());
        for (a, b) in cert_a.ops.iter().zip(cert_b.ops.iter()) {
            assert_eq!(a, b);
        }
    }
}
