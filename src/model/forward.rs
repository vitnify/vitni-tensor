//! Llama2 forward pass — single-token version with KV cache, mirroring
//! `the reference implementation`'s `forward()` function exactly.
//!
//! The KV cache is held as a raw `Vec<f32>` rather than a vitni-tensor
//! Tensor because the cache is fundamentally mutable per-position state,
//! and vitni-tensor's `Arc<Storage>` is intentionally immutable so ops
//! produce fresh tensors. The values flowing THROUGH the cache (k, v
//! computed each step) ARE tensors; the cache itself is the persistent
//! scratch.
//!
//! This is the exact same architectural pattern karpathy uses in
//! llama2.c and the reference implementation uses today.

use super::{config::Config, weights::Weights};
use crate::{error::Result, storage::Storage, tensor::Tensor, Shape};
use alloc::vec::Vec;

/// Persistent state across decode steps. The KV cache lives here.
pub struct RunState {
    /// Key cache — `[n_layers, seq_len, kv_dim]`, flat.
    key_cache: Vec<f32>,
    /// Value cache — `[n_layers, seq_len, kv_dim]`, flat.
    value_cache: Vec<f32>,
}

impl RunState {
    /// Allocate caches sized for the full config.
    pub fn new(cfg: &Config) -> Self {
        let kv_total = cfg.n_layers * cfg.seq_len * cfg.kv_dim();
        Self {
            key_cache: alloc::vec![0.0f32; kv_total],
            value_cache: alloc::vec![0.0f32; kv_total],
        }
    }

    /// Mutable access to the key cache. Used by alternate architecture
    /// forward passes (Gemma, future variants) that need to write
    /// rotated K values back in-place.
    pub fn key_cache_mut(&mut self) -> &mut [f32] {
        &mut self.key_cache
    }

    /// Mutable access to the value cache.
    pub fn value_cache_mut(&mut self) -> &mut [f32] {
        &mut self.value_cache
    }
}

/// Run one decode step: feed `token` at `pos`, return logits over
/// the vocabulary. CPU-only convenience wrapper around
/// `step_with_accel` using `CpuAccelerator`.
pub fn step(
    cfg: &Config,
    weights: &Weights,
    state: &mut RunState,
    token: u32,
    pos: usize,
) -> Result<Tensor> {
    let mut cpu = crate::accel::CpuAccelerator;
    step_with_accel(cfg, weights, state, &mut cpu, token, pos)
}

/// Run one decode step, dispatching matmul/softmax/rms_norm/silu
/// through `accel`. On the host runtime with a `RuntimeGpu` accelerator the
/// large matmuls go through `SYS_GPU_MATMUL`; small ones (and the
/// scoring loop inside attention) stay CPU per the accelerator's
/// policy.
///
/// Mutates the KV cache in `state` to record k/v for this position.
pub fn step_with_accel<A>(
    cfg: &Config,
    weights: &Weights,
    state: &mut RunState,
    accel: &mut A,
    token: u32,
    pos: usize,
) -> core::result::Result<Tensor, A::Error>
where
    A: crate::accel::Accelerator,
{
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim();
    let kv_mul = cfg.kv_mul();
    let head_size = cfg.head_size();
    let hidden_dim = cfg.hidden_dim;
    let n_heads = cfg.n_heads;

    // ---- 0. Token embedding: x = embedding(token, table) ----
    // Single-token shortcut: we know token in advance, so a direct
    // slice copy is the same op as embedding-lookup at 1 token.
    let token_us = token as usize;
    let emb = &weights.token_embedding_table[token_us * dim..(token_us + 1) * dim];
    let mut x = Tensor::from_f32(emb.to_vec(), Shape::new(&[dim])?)?;

    for layer in 0..cfg.n_layers {
        // ---- 1. Attention sublayer ----
        // 1a. Pre-norm
        let rms_w = slice_to_tensor(
            &weights.rms_att_weight[layer * dim..(layer + 1) * dim],
            &[dim],
        )?;
        let xb = accel.rms_norm(&x, &rms_w, 1e-5)?;

        // 1b. Q/K/V projections.
        //
        // llama2.c stores W as `[out_dim, in_dim]` (row i = weights
        // producing output i), so the canonical compute is
        // `out[i] = sum_k x[k] * W[i, k]` — i.e. `out = x @ W^T`.
        //
        // Our `matmul` treats `[a, b] @ [b, c] = [a, c]`. So we
        // pass the weight tensor with shape `[out_dim, in_dim]`,
        // transpose it to `[in_dim, out_dim]`, and matmul against
        // xb `[1, in_dim]` to get `[1, out_dim]`.
        let xb_2d = xb_to_2d(&xb, dim)?;

        let wq_layer = slice_to_tensor(
            &weights.wq[layer * dim * dim..(layer + 1) * dim * dim],
            &[dim, dim],
        )?;
        let wq_t = transpose_2d(&wq_layer, dim, dim)?;
        let q = accel.matmul(&xb_2d, &wq_t)?;

        let wk_layer = slice_to_tensor(
            &weights.wk[layer * dim * kv_dim..(layer + 1) * dim * kv_dim],
            &[kv_dim, dim],
        )?;
        let wk_t = transpose_2d(&wk_layer, kv_dim, dim)?;
        let k = accel.matmul(&xb_2d, &wk_t)?;

        let wv_layer = slice_to_tensor(
            &weights.wv[layer * dim * kv_dim..(layer + 1) * dim * kv_dim],
            &[kv_dim, dim],
        )?;
        let wv_t = transpose_2d(&wv_layer, kv_dim, dim)?;
        let v = accel.matmul(&xb_2d, &wv_t)?;

        // 1c. Store k/v into the cache BEFORE rope (we'll rope k below
        // in-place on the cached slot, matching the reference implementation order).
        let kv_layer_off = layer * cfg.seq_len * kv_dim + pos * kv_dim;
        let k_data = tensor_as_f32(&k)?;
        let v_data = tensor_as_f32(&v)?;
        state.key_cache[kv_layer_off..kv_layer_off + kv_dim].copy_from_slice(k_data);
        state.value_cache[kv_layer_off..kv_layer_off + kv_dim].copy_from_slice(v_data);

        // 1d. RoPE on q (full dim) and the just-stored k (kv_dim).
        // Mirror the reference implementation's per-pair loop exactly so values match.
        let mut q_buf = tensor_as_f32(&q)?.to_vec();
        let mut i = 0;
        while i < dim {
            let head_dim_idx = i % head_size;
            let freq = 1.0f32 / libm::powf(10000.0, head_dim_idx as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = libm::cosf(val);
            let fci = libm::sinf(val);

            let q0 = q_buf[i];
            let q1 = q_buf[i + 1];
            q_buf[i] = q0 * fcr - q1 * fci;
            q_buf[i + 1] = q0 * fci + q1 * fcr;

            if i < kv_dim {
                let k_idx = kv_layer_off + i;
                let k0 = state.key_cache[k_idx];
                let k1 = state.key_cache[k_idx + 1];
                state.key_cache[k_idx] = k0 * fcr - k1 * fci;
                state.key_cache[k_idx + 1] = k0 * fci + k1 * fcr;
            }
            i += 2;
        }

        // 1e. Multi-head attention with KV cache.
        //
        // Per-head: scores = q_h · k_h (across positions 0..=pos)
        //           probs  = softmax(scores / sqrt(head_size))
        //           xb_h   = sum_t probs[t] * v_h[t]
        let mut xb_out = alloc::vec![0.0f32; dim];
        for h in 0..n_heads {
            let q_off = h * head_size;
            let mut att = alloc::vec![0.0f32; pos + 1];
            // Scores
            for t in 0..=pos {
                let k_off =
                    layer * cfg.seq_len * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
                let mut score = 0.0f32;
                for d in 0..head_size {
                    score += q_buf[q_off + d] * state.key_cache[k_off + d];
                }
                score /= libm::sqrtf(head_size as f32);
                att[t] = score;
            }
            // Softmax via the accelerator. Per-head softmax is
            // small (pos+1 elements); a real GPU accel would batch
            // these. For M6 we dispatch one-by-one, the accel can
            // choose CPU fallback for sub-threshold tensors.
            let att_t = Tensor::from_f32(att, Shape::new(&[pos + 1])?)?;
            let probs = accel.softmax_last_dim(&att_t)?;
            let probs_data = tensor_as_f32(&probs)?;

            // Weighted sum of values.
            let xb_off = h * head_size;
            for t in 0..=pos {
                let v_off =
                    layer * cfg.seq_len * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
                let a = probs_data[t];
                for d in 0..head_size {
                    xb_out[xb_off + d] += a * state.value_cache[v_off + d];
                }
            }
        }
        let xb_attn = Tensor::from_f32(xb_out, Shape::new(&[1, dim])?)?;

        // 1f. Output projection. wo is stored [out_dim, in_dim] = [dim, dim].
        let wo_layer = slice_to_tensor(
            &weights.wo[layer * dim * dim..(layer + 1) * dim * dim],
            &[dim, dim],
        )?;
        let wo_t = transpose_2d(&wo_layer, dim, dim)?;
        let xb2 = accel.matmul(&xb_attn, &wo_t)?;
        // Reshape [1, dim] -> [dim] for residual add
        let xb2_1d = squeeze_to_1d(&xb2, dim)?;

        // 1g. Residual.
        x = x.add(&xb2_1d)?;

        // ---- 2. FFN sublayer (SwiGLU) ----
        // 2a. Pre-norm
        let rms_ffn = slice_to_tensor(
            &weights.rms_ffn_weight[layer * dim..(layer + 1) * dim],
            &[dim],
        )?;
        let xb_ffn = accel.rms_norm(&x, &rms_ffn, 1e-5)?;
        let xb_ffn_2d = xb_to_2d(&xb_ffn, dim)?;

        // 2b. Gate and up projections.
        // w1 / w3 are stored [hidden_dim, dim] (out × in). Transpose
        // for matmul against xb_ffn [1, dim] → [1, hidden_dim].
        let w1_layer = slice_to_tensor(
            &weights.w1[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            &[hidden_dim, dim],
        )?;
        let w1_t = transpose_2d(&w1_layer, hidden_dim, dim)?;
        let w3_layer = slice_to_tensor(
            &weights.w3[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            &[hidden_dim, dim],
        )?;
        let w3_t = transpose_2d(&w3_layer, hidden_dim, dim)?;
        let gate = accel.matmul(&xb_ffn_2d, &w1_t)?;
        let up = accel.matmul(&xb_ffn_2d, &w3_t)?;

        // 2c. SwiGLU: silu(gate) * up
        let gate_act = accel.silu(&gate)?;
        let ffn_inner = gate_act.mul(&up)?;

        // 2d. Down projection. w2 stored [dim, hidden_dim] (out × in).
        let w2_layer = slice_to_tensor(
            &weights.w2[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            &[dim, hidden_dim],
        )?;
        let w2_t = transpose_2d(&w2_layer, dim, hidden_dim)?;
        let xb_down = accel.matmul(&ffn_inner, &w2_t)?;
        let xb_down_1d = squeeze_to_1d(&xb_down, dim)?;

        // 2e. Residual.
        x = x.add(&xb_down_1d)?;
    }

    // ---- 3. Final norm + lm_head ----
    let rms_final = slice_to_tensor(weights.rms_final_weight, &[dim])?;
    let x_final = accel.rms_norm(&x, &rms_final, 1e-5)?;
    let x_final_2d = xb_to_2d(&x_final, dim)?;
    let wcls = slice_to_tensor(weights.wcls, &[cfg.vocab_size, dim])?;
    // wcls is stored as [vocab, dim], we need [dim, vocab] for matmul.
    let wcls_t = transpose_2d(&wcls, cfg.vocab_size, dim)?;
    let logits = accel.matmul(&x_final_2d, &wcls_t)?;
    // Reshape [1, vocab] -> [vocab]
    Ok(squeeze_to_1d(&logits, cfg.vocab_size)?)
}

// =============================================================================
// Helpers — small bridges between raw f32 slices and Tensor. These would
// be unnecessary if Tensor had a borrowed-storage constructor; deferred
// to M3 v2 for cleanliness.
// =============================================================================

fn slice_to_tensor(data: &[f32], dims: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(data.to_vec(), Shape::new(dims)?)
}

fn tensor_as_f32(t: &Tensor) -> Result<&[f32]> {
    if let Storage::Cpu(s) = t.storage() {
        Ok(s.as_f32_slice())
    } else {
        Err(crate::error::Error::NotImplemented {
            op: "tensor_as_f32",
            why: "GPU storage not supported in M3 helpers",
        })
    }
}

/// Reshape `[dim]` -> `[1, dim]` for matmul-as-vector.
fn xb_to_2d(t: &Tensor, dim: usize) -> Result<Tensor> {
    let data = tensor_as_f32(t)?.to_vec();
    Tensor::from_f32(data, Shape::new(&[1, dim])?)
}

/// Reshape `[1, n]` or `[n]` -> `[n]`.
fn squeeze_to_1d(t: &Tensor, n: usize) -> Result<Tensor> {
    let data = tensor_as_f32(t)?.to_vec();
    Tensor::from_f32(data, Shape::new(&[n])?)
}

/// Transpose a row-major `[rows, cols]` to `[cols, rows]`.
fn transpose_2d(t: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let src = tensor_as_f32(t)?;
    let mut out = alloc::vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    Tensor::from_f32(out, Shape::new(&[cols, rows])?)
}
