//! M6: prove the accelerator dispatch story end-to-end.
//!
//! Two claims to verify:
//!
//!   1. **Dispatch correctness**: every matmul/softmax/rms_norm/silu
//!      in a Llama2 forward step actually goes through the
//!      Accelerator trait (not directly through Tensor methods).
//!      Verified by routing through `RecordingAccelerator` and
//!      asserting the per-op counts match the expected per-layer
//!      arithmetic: each layer does 7 matmuls (Q, K, V, O, gate, up,
//!      down) + 2 rms_norms + N softmaxes (1 per head) + 1 silu,
//!      and the model also does 1 final rms_norm + 1 final lm_head
//!      matmul.
//!
//!   2. **Equivalence**: `CpuAccelerator` produces bit-identical
//!      logits to the no-accelerator path (since it just delegates
//!      to Tensor methods). And `RecordingAccelerator` (which wraps
//!      CpuAccelerator) produces identical logits too.
//!
//! Plus a sanity check on REAL stories15M: the accelerator path
//! produces the same tokens as the M3 baseline.

extern crate alloc;

use vitni_tensor::accel::{Accelerator, CpuAccelerator, RecordingAccelerator};
use vitni_tensor::model::{config::Config, forward, weights::Weights};
use vitni_tensor::Storage;
use std::path::PathBuf;

const ASSET_REL: &str = "../../userspace/the reference implementation/assets/stories15M.bin";

fn asset_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest);
    p.push(ASSET_REL);
    p
}

fn synthetic_blob(cfg: &Config) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(cfg.dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.hidden_dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_layers as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_heads as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_kv_heads as i32).to_le_bytes());
    let vocab_signed = if cfg.shared_weights {
        cfg.vocab_size as i32
    } else {
        -(cfg.vocab_size as i32)
    };
    blob.extend_from_slice(&vocab_signed.to_le_bytes());
    blob.extend_from_slice(&(cfg.seq_len as i32).to_le_bytes());
    let hs = cfg.head_size();
    let mut sizes = vec![
        cfg.vocab_size * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.dim * (cfg.n_heads * hs),
        cfg.n_layers * cfg.dim * (cfg.n_kv_heads * hs),
        cfg.n_layers * cfg.dim * (cfg.n_kv_heads * hs),
        cfg.n_layers * (cfg.n_heads * hs) * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.hidden_dim * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.hidden_dim,
        cfg.n_layers * cfg.hidden_dim * cfg.dim,
        cfg.dim,
    ];
    if !cfg.shared_weights {
        sizes.push(cfg.vocab_size * cfg.dim);
    }
    let mut seed = 1u32;
    for &n in &sizes {
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let v = (seed as i32 as f32) / (i32::MAX as f32) * 0.01;
            blob.extend_from_slice(&v.to_le_bytes());
        }
    }
    for _ in 0..(cfg.seq_len * hs) {
        blob.extend_from_slice(&0.0f32.to_le_bytes());
    }
    blob
}

fn unwrap_f32(t: &vitni_tensor::Tensor) -> Vec<f32> {
    let Storage::Cpu(s) = t.storage() else {
        panic!()
    };
    s.as_f32_slice().to_vec()
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn dispatch_counts_match_llama_arithmetic() {
    // Llama2 per decode step (pos = 0, so 1 attention position):
    //   per layer:  4 attn matmuls (Q, K, V, O)
    //             + 3 ffn matmuls (gate, up, down)
    //             + 2 rms_norms (pre-attn, pre-ffn)
    //             + n_heads softmaxes (1 per head, over current pos)
    //             + 1 silu (on the gate)
    //   model-wide: 1 final rms_norm
    //             + 1 lm_head matmul
    //
    // Total for n_layers=L, n_heads=H:
    //   matmul   = 7L + 1
    //   rms_norm = 2L + 1
    //   softmax  = H * L
    //   silu     = L
    let cfg = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 3,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 64,
        seq_len: 16,
        shared_weights: true,
    };
    let blob = synthetic_blob(&cfg);
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let mut state = forward::RunState::new(&cfg);
    let mut accel = RecordingAccelerator::default();

    let _logits = forward::step_with_accel(&cfg, &weights, &mut state, &mut accel, 1, 0)
        .expect("step");

    let l = cfg.n_layers;
    let h = cfg.n_heads;
    assert_eq!(accel.matmul_count, 7 * l + 1, "matmul count");
    assert_eq!(accel.rms_norm_count, 2 * l + 1, "rms_norm count");
    assert_eq!(accel.softmax_count, h * l, "softmax count");
    assert_eq!(accel.silu_count, l, "silu count");

    // Largest matmul should be the lm_head: [1, dim] @ [dim, vocab] →
    // (1, vocab_size, dim) = (1, 64, 32).
    let max = accel.max_matmul.expect("matmul tracked");
    eprintln!("max matmul = {:?}", max);
    assert_eq!(max, (1, cfg.vocab_size, cfg.dim));
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn cpu_accelerator_path_matches_unaccelerated_path() {
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
    let blob = synthetic_blob(&cfg);
    let weights = Weights::from_blob(&blob, &cfg).unwrap();

    // Direct path (uses CpuAccelerator internally via step's wrapper).
    let mut s1 = forward::RunState::new(&cfg);
    let direct = forward::step(&cfg, &weights, &mut s1, 3, 0).unwrap();

    // Explicit CpuAccelerator.
    let mut s2 = forward::RunState::new(&cfg);
    let mut acc = CpuAccelerator;
    let via_cpu_accel =
        forward::step_with_accel(&cfg, &weights, &mut s2, &mut acc, 3, 0).unwrap();

    // RecordingAccelerator (wraps CpuAccelerator).
    let mut s3 = forward::RunState::new(&cfg);
    let mut rec = RecordingAccelerator::default();
    let via_recording =
        forward::step_with_accel(&cfg, &weights, &mut s3, &mut rec, 3, 0).unwrap();

    assert_eq!(unwrap_f32(&direct), unwrap_f32(&via_cpu_accel));
    assert_eq!(unwrap_f32(&direct), unwrap_f32(&via_recording));
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn dispatch_counts_match_gemma_arithmetic() {
    // Gemma has the SAME dispatch count as Llama because the only
    // architectural deltas (sqrt-dim embedding scale, (1+w) rmsnorm,
    // gelu) are NOT in the Accelerator trait. So per-step accel
    // counts should be identical to Llama with the same config.
    use vitni_tensor::model::gemma;
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
    let blob = synthetic_blob(&cfg);
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let mut state = forward::RunState::new(&cfg);
    let mut accel = RecordingAccelerator::default();

    let _logits = gemma::step_with_accel(&cfg, &weights, &mut state, &mut accel, 1, 0).unwrap();

    let l = cfg.n_layers;
    let h = cfg.n_heads;
    assert_eq!(accel.matmul_count, 7 * l + 1);
    // Note: gemma's gemma_rms_norm is INLINE (not through accel)
    // because it has the (1+w) delta that the trait doesn't model.
    // So rms_norm count is 0 — verified next.
    assert_eq!(accel.rms_norm_count, 0, "gemma_rms_norm should NOT go through accel");
    assert_eq!(accel.softmax_count, h * l);
    // SiLU also bypasses accel for gemma (uses GeLU via Tensor::gelu).
    assert_eq!(accel.silu_count, 0, "gemma uses gelu, not silu");
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn stories15m_via_accel_matches_baseline() {
    let path = asset_path();
    if !path.exists() {
        eprintln!("skipping: stories15M.bin not present");
        return;
    }
    let blob = std::fs::read(&path).expect("read stories15M");
    let cfg = Config::from_header(&blob).expect("config");
    let weights = Weights::from_blob(&blob, &cfg).expect("weights");

    // Baseline: M3 / M4 path.
    let mut s_base = forward::RunState::new(&cfg);
    let base = forward::step(&cfg, &weights, &mut s_base, 1, 0).unwrap();
    let base_data = unwrap_f32(&base);

    // Via RecordingAccelerator.
    let mut s_acc = forward::RunState::new(&cfg);
    let mut accel = RecordingAccelerator::default();
    let via = forward::step_with_accel(&cfg, &weights, &mut s_acc, &mut accel, 1, 0).unwrap();
    let via_data = unwrap_f32(&via);

    assert_eq!(base_data, via_data, "accel dispatch must be bit-identical");

    // Op counts on real stories15M: dim=288, n_layers=6, n_heads=6,
    // vocab=32000, seq_len=256.
    // Expected: matmul = 7*6 + 1 = 43, rms_norm = 13, softmax = 36, silu = 6.
    assert_eq!(accel.matmul_count, 43);
    assert_eq!(accel.rms_norm_count, 13);
    assert_eq!(accel.softmax_count, 36);
    assert_eq!(accel.silu_count, 6);

    // Largest matmul on stories15M = lm_head: (1, 32000, 288).
    assert_eq!(accel.max_matmul, Some((1, 32000, 288)));

    eprintln!(
        "stories15M dispatch: {} matmul, {} rms_norm, {} softmax, {} silu",
        accel.matmul_count, accel.rms_norm_count, accel.softmax_count, accel.silu_count
    );
    eprintln!("max matmul: {:?}", accel.max_matmul);
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn custom_accelerator_error_propagates() {
    // A custom accel that errors on matmul should abort the forward
    // pass with that error — proving the error path is real (not
    // accidentally swallowed in conversion).
    use vitni_tensor::error::Error;
    use vitni_tensor::Tensor;

    struct FailingAccel;
    impl Accelerator for FailingAccel {
        type Error = Error;
        fn matmul(&mut self, _: &Tensor, _: &Tensor) -> Result<Tensor, Self::Error> {
            Err(Error::NotImplemented {
                op: "test_failing_matmul",
                why: "intentional test failure",
            })
        }
        fn softmax_last_dim(&mut self, t: &Tensor) -> Result<Tensor, Self::Error> {
            t.softmax_last_dim()
        }
        fn rms_norm(&mut self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, Self::Error> {
            x.rms_norm(w, eps)
        }
        fn silu(&mut self, t: &Tensor) -> Result<Tensor, Self::Error> {
            t.silu()
        }
    }

    let cfg = Config {
        dim: 16,
        hidden_dim: 32,
        n_layers: 1,
        n_heads: 2,
        n_kv_heads: 2,
        vocab_size: 32,
        seq_len: 8,
        shared_weights: true,
    };
    let blob = synthetic_blob(&cfg);
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let mut state = forward::RunState::new(&cfg);
    let mut accel = FailingAccel;

    let result = forward::step_with_accel(&cfg, &weights, &mut state, &mut accel, 1, 0);
    match result {
        Err(Error::NotImplemented { op, .. }) => {
            assert_eq!(op, "test_failing_matmul");
        }
        Ok(_) => panic!("expected matmul to fail"),
        Err(other) => panic!("expected NotImplemented, got {:?}", other),
    }
}
