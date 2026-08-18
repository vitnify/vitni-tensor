//! Gemma forward pass — second architecture port.
//!
//! Architecturally identical to Llama2 except for three deltas:
//!
//!   1. **RMSNorm scaling**: `output = x * scale * (1.0 + weight)`
//!      instead of `output = x * scale * weight`. Gemma's training
//!      treats the weight as a residual on top of 1.0.
//!
//!   2. **Embedding scaling**: at the input, the token embedding is
//!      multiplied by `sqrt(dim)`. This compensates for Gemma's tied
//!      embedding tables (the lm_head shares weights with the input
//!      embedding, so without the scale the logits would be tiny).
//!
//!   3. **FFN activation**: `GeLU(gate) * up` instead of Llama's
//!      `SiLU(gate) * up`. Uses the existing `Tensor::gelu` op.
//!
//! Everything else — RoPE, multi-head attention, KV cache layout,
//! residual structure — matches `forward::step` exactly. This is the
//! "tiny architectural delta = tiny code delta" demonstration that
//! M5 sets out to prove. The diff against `forward::step` is ~30 LOC.

use super::{config::Config, forward::RunState, weights::Weights};
use crate::{error::Result, storage::Storage, tensor::Tensor, Shape};

/// One decode step under the Gemma architecture. Returns logits over
/// the vocabulary.
///
/// API mirrors `forward::step` so callers can switch architecture
/// with a one-line change. CPU-only wrapper around
/// `step_with_accel`.
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

/// Gemma step with explicit accelerator dispatch.
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

    // ---- 0. Token embedding + Gemma scaling ----
    // The ONLY Gemma-vs-Llama difference at the input.
    let token_us = token as usize;
    let emb = &weights.token_embedding_table[token_us * dim..(token_us + 1) * dim];
    let scale = libm::sqrtf(dim as f32);
    let scaled: alloc::vec::Vec<f32> = emb.iter().map(|v| v * scale).collect();
    let mut x = Tensor::from_f32(scaled, Shape::new(&[dim])?)?;

    for layer in 0..cfg.n_layers {
        // ---- 1. Attention sublayer ----
        let rms_w = slice_to_tensor(
            &weights.rms_att_weight[layer * dim..(layer + 1) * dim],
            &[dim],
        )?;
        let xb = gemma_rms_norm(&x, &rms_w, 1e-6)?; // Gemma uses 1e-6 + (1+w)
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

        let kv_layer_off = layer * cfg.seq_len * kv_dim + pos * kv_dim;
        let k_data = tensor_as_f32(&k)?;
        let v_data = tensor_as_f32(&v)?;
        state_kc_mut(state)[kv_layer_off..kv_layer_off + kv_dim].copy_from_slice(k_data);
        state_vc_mut(state)[kv_layer_off..kv_layer_off + kv_dim].copy_from_slice(v_data);

        // RoPE — identical to Llama. Gemma 1 uses theta=10000.
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
                let k0 = state_kc_mut(state)[k_idx];
                let k1 = state_kc_mut(state)[k_idx + 1];
                state_kc_mut(state)[k_idx] = k0 * fcr - k1 * fci;
                state_kc_mut(state)[k_idx + 1] = k0 * fci + k1 * fcr;
            }
            i += 2;
        }

        // Multi-head attention with KV cache.
        let mut xb_out = alloc::vec![0.0f32; dim];
        for h in 0..n_heads {
            let q_off = h * head_size;
            let mut att = alloc::vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let k_off =
                    layer * cfg.seq_len * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
                let mut score = 0.0f32;
                for d in 0..head_size {
                    score += q_buf[q_off + d] * state_kc_mut(state)[k_off + d];
                }
                score /= libm::sqrtf(head_size as f32);
                att[t] = score;
            }
            let att_t = Tensor::from_f32(att, Shape::new(&[pos + 1])?)?;
            let probs = accel.softmax_last_dim(&att_t)?;
            let probs_data = tensor_as_f32(&probs)?;
            let xb_off = h * head_size;
            for t in 0..=pos {
                let v_off =
                    layer * cfg.seq_len * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
                let a = probs_data[t];
                for d in 0..head_size {
                    xb_out[xb_off + d] += a * state_vc_mut(state)[v_off + d];
                }
            }
        }
        let xb_attn = Tensor::from_f32(xb_out, Shape::new(&[1, dim])?)?;
        let wo_layer = slice_to_tensor(
            &weights.wo[layer * dim * dim..(layer + 1) * dim * dim],
            &[dim, dim],
        )?;
        let wo_t = transpose_2d(&wo_layer, dim, dim)?;
        let xb2 = accel.matmul(&xb_attn, &wo_t)?;
        let xb2_1d = squeeze_to_1d(&xb2, dim)?;
        x = x.add(&xb2_1d)?;

        // ---- 2. FFN sublayer (GeGLU, not SwiGLU) ----
        let rms_ffn = slice_to_tensor(
            &weights.rms_ffn_weight[layer * dim..(layer + 1) * dim],
            &[dim],
        )?;
        let xb_ffn = gemma_rms_norm(&x, &rms_ffn, 1e-6)?;
        let xb_ffn_2d = xb_to_2d(&xb_ffn, dim)?;

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

        // Gemma delta #3: GeLU not SiLU. GeLU not in the Accelerator
        // trait (Llama-family doesn't need it); ops::unary::gelu via
        // Tensor stays the CPU path. A future Accelerator extension
        // can add it if Gemma performance becomes GPU-bound.
        let gate_act = gate.gelu()?;
        let ffn_inner = gate_act.mul(&up)?;

        let w2_layer = slice_to_tensor(
            &weights.w2[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            &[dim, hidden_dim],
        )?;
        let w2_t = transpose_2d(&w2_layer, dim, hidden_dim)?;
        let xb_down = accel.matmul(&ffn_inner, &w2_t)?;
        let xb_down_1d = squeeze_to_1d(&xb_down, dim)?;
        x = x.add(&xb_down_1d)?;
    }

    // ---- 3. Final norm + lm_head ----
    let rms_final = slice_to_tensor(weights.rms_final_weight, &[dim])?;
    let x_final = gemma_rms_norm(&x, &rms_final, 1e-6)?;
    let x_final_2d = xb_to_2d(&x_final, dim)?;
    let wcls = slice_to_tensor(weights.wcls, &[cfg.vocab_size, dim])?;
    let wcls_t = transpose_2d(&wcls, cfg.vocab_size, dim)?;
    let logits = accel.matmul(&x_final_2d, &wcls_t)?;
    Ok(squeeze_to_1d(&logits, cfg.vocab_size)?)
}

/// Gemma RMSNorm: scale by `(1.0 + weight)` instead of `weight`.
/// `output[i] = x[i] / sqrt(mean(x^2) + eps) * (1.0 + w[i])`.
///
/// Implemented inline (rather than as a new op variant) so the
/// architectural delta is visible at the call site.
fn gemma_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let xs = tensor_as_f32(x)?;
    let ws = tensor_as_f32(weight)?;
    let dims = x.shape().dims();
    let feat = *dims.last().unwrap();
    let rows = xs.len() / feat;
    let mut out = alloc::vec![0.0f32; xs.len()];
    for r in 0..rows {
        let row = &xs[r * feat..(r + 1) * feat];
        let mut sumsq = 0.0f32;
        for &v in row {
            sumsq += v * v;
        }
        let scale = 1.0 / libm::sqrtf(sumsq / feat as f32 + eps);
        let dst = &mut out[r * feat..(r + 1) * feat];
        for (i, &v) in row.iter().enumerate() {
            dst[i] = v * scale * (1.0 + ws[i]); // <-- Gemma delta
        }
    }
    Tensor::from_f32(out, *x.shape())
}

// ---- helpers shared with forward.rs (duplicated to keep modules
// self-contained; we'll consolidate when a third architecture lands) ----

fn slice_to_tensor(data: &[f32], dims: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(data.to_vec(), Shape::new(dims)?)
}

fn tensor_as_f32(t: &Tensor) -> Result<&[f32]> {
    if let Storage::Cpu(s) = t.storage() {
        Ok(s.as_f32_slice())
    } else {
        Err(crate::error::Error::NotImplemented {
            op: "tensor_as_f32",
            why: "GPU storage not supported",
        })
    }
}

fn xb_to_2d(t: &Tensor, dim: usize) -> Result<Tensor> {
    let data = tensor_as_f32(t)?.to_vec();
    Tensor::from_f32(data, Shape::new(&[1, dim])?)
}

fn squeeze_to_1d(t: &Tensor, n: usize) -> Result<Tensor> {
    let data = tensor_as_f32(t)?.to_vec();
    Tensor::from_f32(data, Shape::new(&[n])?)
}

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

// RunState's KV cache fields are private; we need mutable access.
// Add accessor helpers — these belong on RunState eventually.
fn state_kc_mut(state: &mut RunState) -> &mut [f32] {
    state.key_cache_mut()
}
fn state_vc_mut(state: &mut RunState) -> &mut [f32] {
    state.value_cache_mut()
}
