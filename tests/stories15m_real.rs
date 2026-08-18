//! M3 stretch goal: cross-verify vitni-tensor vs reference on REAL
//! stories15M weights (60 MB binary from karpathy's llama2.c, the
//! same blob the reference implementation runs).
//!
//! This is the host-side proof that the architecture is correct at
//! production scale (dim=288, n_layers=6, vocab=32000, etc.), not
//! just on the shrunken synthetic config from `llama2_reference.rs`.
//!
//! Skips gracefully if the asset blob is absent so CI on a fresh
//! checkout doesn't break.

extern crate alloc;

use vitni_tensor::model::{config::Config, forward, weights::Weights};
use vitni_tensor::Storage;
use std::path::PathBuf;

const ASSET_REL: &str = "../../userspace/the reference implementation/assets/stories15M.bin";

fn asset_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let mut p = PathBuf::from(manifest);
    p.push(ASSET_REL);
    p
}

fn ref_rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let scale = 1.0 / (sumsq / n as f32 + eps).sqrt();
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
        *v = (*v - mx).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

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
    kc: &mut [f32],
    vc: &mut [f32],
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let dim = cfg.dim;
    let kv_dim = cfg.kv_dim();
    let kv_mul = cfg.kv_mul();
    let head_size = cfg.head_size();
    let hidden_dim = cfg.hidden_dim;
    let n_heads = cfg.n_heads;

    let mut x: Vec<f32> = weights.token_embedding_table
        [(token as usize) * dim..((token as usize) + 1) * dim]
        .to_vec();
    let mut xb = vec![0.0f32; dim];
    let mut xb2 = vec![0.0f32; dim];
    let mut q = vec![0.0f32; dim];
    let mut hb = vec![0.0f32; hidden_dim];
    let mut hb2 = vec![0.0f32; hidden_dim];

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
        let mut ks = vec![0.0f32; kv_dim];
        ref_matmul(
            &mut ks,
            &xb,
            &weights.wk[layer * dim * kv_dim..(layer + 1) * dim * kv_dim],
            dim,
            kv_dim,
        );
        kc[kv_off..kv_off + kv_dim].copy_from_slice(&ks);
        let mut vs = vec![0.0f32; kv_dim];
        ref_matmul(
            &mut vs,
            &xb,
            &weights.wv[layer * dim * kv_dim..(layer + 1) * dim * kv_dim],
            dim,
            kv_dim,
        );
        vc[kv_off..kv_off + kv_dim].copy_from_slice(&vs);

        let mut i = 0;
        while i < dim {
            let head_dim_idx = i % head_size;
            let freq = 1.0 / 10000.0_f32.powf(head_dim_idx as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let q0 = q[i];
            let q1 = q[i + 1];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            if i < kv_dim {
                let kx = kv_off + i;
                let k0 = kc[kx];
                let k1 = kc[kx + 1];
                kc[kx] = k0 * fcr - k1 * fci;
                kc[kx + 1] = k0 * fci + k1 * fcr;
            }
            i += 2;
        }

        for h in 0..n_heads {
            let q_off = h * head_size;
            let mut att = vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let k_off = layer * cfg.seq_len * kv_dim
                    + t * kv_dim
                    + (h / kv_mul) * head_size;
                let mut score = 0.0f32;
                for d in 0..head_size {
                    score += q[q_off + d] * kc[k_off + d];
                }
                att[t] = score / (head_size as f32).sqrt();
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
                    xb[xb_off + d] += a * vc[v_off + d];
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
            hb[d] = v * (1.0 / (1.0 + (-v).exp())) * hb2[d];
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
    let mut x_final = vec![0.0f32; dim];
    ref_rmsnorm(&mut x_final, &x_in, weights.rms_final_weight, 1e-5);
    let mut logits = vec![0.0f32; cfg.vocab_size];
    ref_matmul(&mut logits, &x_final, weights.wcls, dim, cfg.vocab_size);
    logits
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn stories15m_real_weights_greedy_decode_matches() {
    let path = asset_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let blob = std::fs::read(&path).expect("read stories15M.bin");
    eprintln!("loaded stories15M: {} bytes", blob.len());

    let cfg = Config::from_header(&blob).expect("parse config");
    eprintln!(
        "cfg: dim={} hidden_dim={} n_layers={} n_heads={} n_kv_heads={} vocab={} seq_len={} shared={}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
        cfg.vocab_size, cfg.seq_len, cfg.shared_weights
    );
    assert_eq!(cfg.dim, 288);
    assert_eq!(cfg.n_layers, 6);
    assert_eq!(cfg.vocab_size, 32000);

    let weights = Weights::from_blob(&blob, &cfg).expect("parse weights");

    // Walk a handful of positions. Use token 1 ("<s>"-ish — value doesn't
    // matter for the equivalence test, just needs to be < vocab) and
    // continue greedily for N steps.
    const N_STEPS: usize = 5;
    let start_token: u32 = 1;

    // Reference path
    let mut ref_kc = vec![0.0f32; cfg.n_layers * cfg.seq_len * cfg.kv_dim()];
    let mut ref_vc = vec![0.0f32; cfg.n_layers * cfg.seq_len * cfg.kv_dim()];
    let mut ref_tokens: Vec<u32> = Vec::new();
    let mut cur = start_token;
    for pos in 0..N_STEPS {
        let logits = ref_forward(&cfg, &weights, &mut ref_kc, &mut ref_vc, cur, pos);
        let mut top = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                top = i;
            }
        }
        cur = top as u32;
        ref_tokens.push(cur);
    }
    eprintln!("reference tokens: {:?}", ref_tokens);

    // vitni-tensor path
    let mut state = forward::RunState::new(&cfg);
    let mut dt_tokens: Vec<u32> = Vec::new();
    let mut cur = start_token;
    for pos in 0..N_STEPS {
        let t = forward::step(&cfg, &weights, &mut state, cur, pos)
            .expect("vitni-tensor step");
        let argmax = t.argmax_last_dim().expect("argmax");
        let Storage::Cpu(s) = argmax.storage() else {
            panic!()
        };
        let top = u32::from_le_bytes(s.as_bytes()[..4].try_into().unwrap());
        cur = top;
        dt_tokens.push(cur);
    }
    eprintln!("vitni-tensor tokens: {:?}", dt_tokens);

    assert_eq!(
        ref_tokens, dt_tokens,
        "Token sequences diverge between reference and vitni-tensor on REAL stories15M weights"
    );
}
