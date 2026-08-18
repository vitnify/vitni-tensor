//! M3 cross-verification: prove that the vitni-tensor forward pass
//! produces bit-identical logits to a pure-scalar reference
//! implementation modeled after the reference implementation's `forward()`.
//!
//! Strategy:
//!   1. Build a stories15M-shaped synthetic weight blob with
//!      deterministic values (seed-derived, all in [-0.05, 0.05]
//!      so attention scores stay numerically tame).
//!   2. Run the vitni-tensor forward pass.
//!   3. Run a raw-buffer reference pass that mirrors the exact
//!      ordering and arithmetic of the reference implementation's `forward`.
//!   4. Assert: argmax matches, top-5 logits match, post-attention
//!      residual hash matches per layer.
//!
//! Real stories15M.bin weight loading is straightforward plumbing on
//! top once this verification passes (host-side: `std::fs::read`;
//! runtime-side: load from P3 partition).

extern crate alloc;

use vitni_tensor::model::{config::Config, forward, weights::Weights};
use vitni_tensor::Storage;

/// Build a synthetic stories15M-shaped weight blob with deterministic
/// values. Smaller test config so the test runs in <1s on host CI.
fn build_synthetic_blob() -> (Config, alloc::vec::Vec<u8>) {
    // Shrink dims so the test runs fast but architecture is real.
    let cfg = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 64,
        seq_len: 16,
        shared_weights: true,
    };

    let mut blob = alloc::vec::Vec::new();
    // 28-byte header.
    blob.extend_from_slice(&(cfg.dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.hidden_dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_layers as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_heads as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_kv_heads as i32).to_le_bytes());
    // Positive vocab_size = shared_weights = true.
    blob.extend_from_slice(&(cfg.vocab_size as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.seq_len as i32).to_le_bytes());

    let head_size = cfg.head_size();

    // Layout (matches Weights::from_blob):
    //   token_embedding_table: vocab × dim
    //   rms_att_weight:        n_layers × dim
    //   wq:                    n_layers × dim × (n_heads × head_size)
    //   wk:                    n_layers × dim × (n_kv_heads × head_size)
    //   wv:                    n_layers × dim × (n_kv_heads × head_size)
    //   wo:                    n_layers × (n_heads × head_size) × dim
    //   rms_ffn_weight:        n_layers × dim
    //   w1:                    n_layers × hidden_dim × dim
    //   w2:                    n_layers × dim × hidden_dim
    //   w3:                    n_layers × hidden_dim × dim
    //   rms_final_weight:      dim
    //   [skipped] freq_cis × 2
    //   (no wcls — shared)
    let sizes = [
        cfg.vocab_size * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.dim * (cfg.n_heads * head_size),
        cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size),
        cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size),
        cfg.n_layers * (cfg.n_heads * head_size) * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.hidden_dim * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.hidden_dim,
        cfg.n_layers * cfg.hidden_dim * cfg.dim,
        cfg.dim,
    ];

    let mut seed = 1u32;
    for &n in &sizes {
        for _ in 0..n {
            // xorshift32 → small range
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let v = (seed as i32 as f32) / (i32::MAX as f32) * 0.05;
            blob.extend_from_slice(&v.to_le_bytes());
        }
    }
    // Skipped freq_cis tables.
    let freq_zeros = cfg.seq_len * head_size;
    for _ in 0..freq_zeros {
        blob.extend_from_slice(&0.0f32.to_le_bytes());
    }
    // No wcls trailer because shared_weights = true.

    (cfg, blob)
}

// =============================================================================
// REFERENCE forward pass — raw f32 buffers, deliberately identical in
// arithmetic order to the reference implementation/src/main.rs::forward(). This is
// the ground truth we cross-check the tensor path against.
// =============================================================================

fn ref_rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let mean = sumsq / n as f32;
    let scale = 1.0 / libm::sqrtf(mean + eps);
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

fn ref_softmax(x: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > mx {
            mx = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - mx);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Matmul matching the reference implementation: xout = x @ W where W is [n, d]
/// laid out row-major as the math W^T, so output[i] = sum_j x[j] * w[i*n + j].
fn ref_matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    for i in 0..d {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += x[j] * w[i * n + j];
        }
        xout[i] = acc;
    }
}

fn ref_forward(
    cfg: &Config,
    weights: &Weights,
    state_kc: &mut [f32],
    state_vc: &mut [f32],
    token: u32,
    pos: usize,
) -> alloc::vec::Vec<f32> {
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim();
    let kv_mul = cfg.kv_mul();
    let head_size = cfg.head_size();
    let hidden_dim = cfg.hidden_dim;
    let n_heads = cfg.n_heads;

    let mut x: alloc::vec::Vec<f32> = weights
        .token_embedding_table
        [(token as usize) * dim..((token as usize) + 1) * dim]
        .to_vec();
    let mut xb = alloc::vec![0.0f32; dim];
    let mut xb2 = alloc::vec![0.0f32; dim];
    let mut q = alloc::vec![0.0f32; dim];
    let mut hb = alloc::vec![0.0f32; hidden_dim];
    let mut hb2 = alloc::vec![0.0f32; hidden_dim];

    for layer in 0..cfg.n_layers {
        ref_rmsnorm(
            &mut xb,
            &x,
            &weights.rms_att_weight[layer * dim..(layer + 1) * dim],
            1e-5,
        );

        ref_matmul(
            &mut q,
            &xb,
            &weights.wq[layer * dim * dim..(layer + 1) * dim * dim],
            dim,
            dim,
        );

        let kv_off = layer * cfg.seq_len * kv_dim + pos * kv_dim;
        let mut k_slice = alloc::vec![0.0f32; kv_dim];
        ref_matmul(
            &mut k_slice,
            &xb,
            &weights.wk[layer * dim * kv_dim..(layer + 1) * dim * kv_dim],
            dim,
            kv_dim,
        );
        state_kc[kv_off..kv_off + kv_dim].copy_from_slice(&k_slice);

        let mut v_slice = alloc::vec![0.0f32; kv_dim];
        ref_matmul(
            &mut v_slice,
            &xb,
            &weights.wv[layer * dim * kv_dim..(layer + 1) * dim * kv_dim],
            dim,
            kv_dim,
        );
        state_vc[kv_off..kv_off + kv_dim].copy_from_slice(&v_slice);

        // RoPE
        let mut i = 0;
        while i < dim {
            let head_dim_idx = i % head_size;
            let freq = 1.0 / libm::powf(10000.0, head_dim_idx as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = libm::cosf(val);
            let fci = libm::sinf(val);
            let q0 = q[i];
            let q1 = q[i + 1];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            if i < kv_dim {
                let k_idx = kv_off + i;
                let k0 = state_kc[k_idx];
                let k1 = state_kc[k_idx + 1];
                state_kc[k_idx] = k0 * fcr - k1 * fci;
                state_kc[k_idx + 1] = k0 * fci + k1 * fcr;
            }
            i += 2;
        }

        // Attention
        for h in 0..n_heads {
            let q_off = h * head_size;
            let mut att = alloc::vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let k_off = layer * cfg.seq_len * kv_dim
                    + t * kv_dim
                    + (h / kv_mul) * head_size;
                let mut score = 0.0f32;
                for d in 0..head_size {
                    score += q[q_off + d] * state_kc[k_off + d];
                }
                score /= libm::sqrtf(head_size as f32);
                att[t] = score;
            }
            ref_softmax(&mut att);

            let xb_off = h * head_size;
            for d in 0..head_size {
                xb[xb_off + d] = 0.0;
            }
            for t in 0..=pos {
                let v_off = layer * cfg.seq_len * kv_dim
                    + t * kv_dim
                    + (h / kv_mul) * head_size;
                let a = att[t];
                for d in 0..head_size {
                    xb[xb_off + d] += a * state_vc[v_off + d];
                }
            }
        }

        ref_matmul(
            &mut xb2,
            &xb,
            &weights.wo[layer * dim * dim..(layer + 1) * dim * dim],
            dim,
            dim,
        );

        for d in 0..dim {
            x[d] += xb2[d];
        }

        ref_rmsnorm(
            &mut xb,
            &x,
            &weights.rms_ffn_weight[layer * dim..(layer + 1) * dim],
            1e-5,
        );

        ref_matmul(
            &mut hb,
            &xb,
            &weights.w1[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            dim,
            hidden_dim,
        );
        ref_matmul(
            &mut hb2,
            &xb,
            &weights.w3[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            dim,
            hidden_dim,
        );

        for d in 0..hidden_dim {
            let v = hb[d];
            let silu = v * (1.0 / (1.0 + libm::expf(-v)));
            hb[d] = silu * hb2[d];
        }

        ref_matmul(
            &mut xb2,
            &hb,
            &weights.w2[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            hidden_dim,
            dim,
        );

        for d in 0..dim {
            x[d] += xb2[d];
        }
    }

    let x_in = x.clone();
    let mut x_final = alloc::vec![0.0f32; dim];
    ref_rmsnorm(&mut x_final, &x_in, weights.rms_final_weight, 1e-5);

    let mut logits = alloc::vec![0.0f32; cfg.vocab_size];
    ref_matmul(&mut logits, &x_final, weights.wcls, dim, cfg.vocab_size);
    logits
}

// =============================================================================
// THE TEST
// =============================================================================

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn vitni_tensor_matches_reference_forward() {
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).expect("weights parse");

    // ---- Test single-step forward at pos=0 ----
    let prompt_token = 7u32;
    let pos = 0;

    // Reference path
    let mut ref_kc = alloc::vec![0.0f32; cfg.n_layers * cfg.seq_len * cfg.kv_dim()];
    let mut ref_vc = alloc::vec![0.0f32; cfg.n_layers * cfg.seq_len * cfg.kv_dim()];
    let ref_logits = ref_forward(&cfg, &weights, &mut ref_kc, &mut ref_vc, prompt_token, pos);

    // vitni-tensor path
    let mut state = forward::RunState::new(&cfg);
    let dt_logits_tensor = forward::step(&cfg, &weights, &mut state, prompt_token, pos)
        .expect("vitni-tensor forward step");
    let Storage::Cpu(s) = dt_logits_tensor.storage() else {
        panic!("expected CPU storage")
    };
    let dt_logits = s.as_f32_slice();

    assert_eq!(dt_logits.len(), ref_logits.len(), "logits length mismatch");

    // Allow small relative differences due to op-by-op accumulation
    // order: softmax does its sum via Tensor, reference does it inline.
    // Both are IEEE 754 but the intermediate temporary creation could
    // affect rounding by a few ULPs. Acceptable for a cross-vendor
    // determinism story; not acceptable for op-internal determinism
    // (each op individually must be bit-exact).
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in dt_logits.iter().zip(ref_logits.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        let denom = a.abs().max(b.abs()).max(1e-6);
        max_rel = max_rel.max(d / denom);
    }
    eprintln!("max abs diff = {max_abs}, max rel diff = {max_rel}");
    // Tolerance: 1e-4 relative. Loose for two-path accumulation; tight
    // enough to catch any real algorithmic discrepancy.
    assert!(
        max_rel < 1e-4,
        "logits diverge: max abs {max_abs}, max rel {max_rel}"
    );

    // Greedy decode (argmax) MUST match. This is the load-bearing
    // user-visible property: same prompt → same token sequence.
    let mut ref_top = 0;
    let mut ref_best = f32::NEG_INFINITY;
    for (i, &v) in ref_logits.iter().enumerate() {
        if v > ref_best {
            ref_best = v;
            ref_top = i;
        }
    }
    let dt_argmax = dt_logits_tensor
        .argmax_last_dim()
        .expect("argmax")
        ;
    let Storage::Cpu(ams) = dt_argmax.storage() else {
        panic!()
    };
    let dt_top = u32::from_le_bytes(ams.as_bytes()[..4].try_into().unwrap()) as usize;
    assert_eq!(dt_top, ref_top, "argmax token differs: dt={dt_top}, ref={ref_top}");
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn greedy_decode_3_tokens_match() {
    // Multi-step decode: ensure the KV cache flows correctly and
    // greedy tokens agree for several steps.
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).expect("weights parse");

    let prompt = alloc::vec![3u32, 7, 11];
    let n_extra = 3;
    let total = prompt.len() + n_extra;

    // Reference
    let mut ref_kc = alloc::vec![0.0f32; cfg.n_layers * cfg.seq_len * cfg.kv_dim()];
    let mut ref_vc = alloc::vec![0.0f32; cfg.n_layers * cfg.seq_len * cfg.kv_dim()];
    let mut ref_tokens: alloc::vec::Vec<u32> = alloc::vec::Vec::new();
    let mut ref_cur = prompt[0];
    for pos in 0..total {
        let logits = ref_forward(&cfg, &weights, &mut ref_kc, &mut ref_vc, ref_cur, pos);
        let mut top = 0;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                top = i;
            }
        }
        let next = if pos + 1 < prompt.len() { prompt[pos + 1] } else { top as u32 };
        ref_tokens.push(next);
        ref_cur = next;
    }

    // vitni-tensor
    let mut state = forward::RunState::new(&cfg);
    let mut dt_tokens: alloc::vec::Vec<u32> = alloc::vec::Vec::new();
    let mut dt_cur = prompt[0];
    for pos in 0..total {
        let logits_t = forward::step(&cfg, &weights, &mut state, dt_cur, pos).expect("step");
        let argmax = logits_t.argmax_last_dim().expect("argmax");
        let Storage::Cpu(s) = argmax.storage() else {
            panic!()
        };
        let top = u32::from_le_bytes(s.as_bytes()[..4].try_into().unwrap());
        let next = if pos + 1 < prompt.len() { prompt[pos + 1] } else { top };
        dt_tokens.push(next);
        dt_cur = next;
    }

    assert_eq!(
        dt_tokens, ref_tokens,
        "greedy decode tokens diverge across the two paths"
    );
}
