//! M4 capstone: ExCert per inference.
//!
//! Two stories:
//!
//! 1. **Software cert determinism + binding**: running the same
//!    inference twice produces identical certs (the load-bearing
//!    determinism property). Changing ANY input — model_id, weights
//!    hash, prompt, n_new_tokens — changes the digest.
//!
//! 2. **Real-weights cert + replay**: with the actual stories15M
//!    weights, produce a cert. Then a "verifier" (us, in the same
//!    test) re-runs from scratch with the same inputs and confirms
//!    it gets the same cert digest. This is the verifiable-inference
//!    claim: anyone with the same inputs can independently
//!    reconstruct the cert and check.

extern crate alloc;

use vitni_tensor::model::{config::Config, inference, weights::Weights};
use std::path::PathBuf;

const ASSET_REL: &str = "../../userspace/the reference implementation/assets/stories15M.bin";

fn asset_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let mut p = PathBuf::from(manifest);
    p.push(ASSET_REL);
    p
}

/// Build a tiny synthetic weights blob matching the layout. Same
/// generator as `llama2_reference.rs::build_synthetic_blob` —
/// minimal duplication, but cleaner than a shared helper that adds
/// crate surface.
fn build_synthetic_blob() -> (Config, Vec<u8>) {
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
    let mut blob = Vec::new();
    blob.extend_from_slice(&(cfg.dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.hidden_dim as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_layers as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_heads as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.n_kv_heads as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.vocab_size as i32).to_le_bytes());
    blob.extend_from_slice(&(cfg.seq_len as i32).to_le_bytes());

    let head_size = cfg.head_size();
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
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let v = (seed as i32 as f32) / (i32::MAX as f32) * 0.05;
            blob.extend_from_slice(&v.to_le_bytes());
        }
    }
    let freq_zeros = cfg.seq_len * head_size;
    for _ in 0..freq_zeros {
        blob.extend_from_slice(&0.0f32.to_le_bytes());
    }
    (cfg, blob)
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn synthetic_cert_determinism() {
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let weights_hash = *blake3::hash(&blob).as_bytes();

    let req = inference::Request {
        model_id: "synthetic-test",
        prompt_tokens: &[3, 7, 11],
        n_new_tokens: 3,
    };

    let a = inference::run(&cfg, &weights, &weights_hash, &req).unwrap();
    let b = inference::run(&cfg, &weights, &weights_hash, &req).unwrap();

    assert_eq!(a.generated_tokens, b.generated_tokens);
    assert_eq!(a.cert.digest, b.cert.digest);
    assert_eq!(a.cert.digest_hex().len(), 64);
    eprintln!("digest = {}", a.cert.digest_hex());
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn cert_binding_changes_with_input() {
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let weights_hash = *blake3::hash(&blob).as_bytes();

    let base = inference::Request {
        model_id: "m",
        prompt_tokens: &[3, 7, 11],
        n_new_tokens: 2,
    };
    let base_digest = inference::run(&cfg, &weights, &weights_hash, &base).unwrap().cert.digest;

    // Different prompt
    let diff_prompt = inference::Request {
        model_id: "m",
        prompt_tokens: &[3, 7, 12],
        n_new_tokens: 2,
    };
    let d1 = inference::run(&cfg, &weights, &weights_hash, &diff_prompt).unwrap().cert.digest;
    assert_ne!(base_digest, d1, "different prompt should change cert");

    // Different model_id
    let diff_id = inference::Request {
        model_id: "other",
        prompt_tokens: &[3, 7, 11],
        n_new_tokens: 2,
    };
    let d2 = inference::run(&cfg, &weights, &weights_hash, &diff_id).unwrap().cert.digest;
    assert_ne!(base_digest, d2, "different model_id should change cert");

    // Different weights_hash (forged)
    let bogus_hash = [0xffu8; 32];
    let d3 = inference::run(&cfg, &weights, &bogus_hash, &base).unwrap().cert.digest;
    assert_ne!(base_digest, d3, "different weights_hash should change cert");

    // Different n_new_tokens (still produces a different cert even
    // though the first N tokens overlap)
    let diff_n = inference::Request {
        model_id: "m",
        prompt_tokens: &[3, 7, 11],
        n_new_tokens: 3,
    };
    let d4 = inference::run(&cfg, &weights, &weights_hash, &diff_n).unwrap().cert.digest;
    assert_ne!(base_digest, d4, "different n_new_tokens should change cert");
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn cert_contains_expected_fields() {
    let (cfg, blob) = build_synthetic_blob();
    let weights = Weights::from_blob(&blob, &cfg).unwrap();
    let weights_hash = *blake3::hash(&blob).as_bytes();
    let req = inference::Request {
        model_id: "stories15M-llama2c",
        prompt_tokens: &[1],
        n_new_tokens: 2,
    };
    let out = inference::run(&cfg, &weights, &weights_hash, &req).unwrap();

    // Inputs we required:
    assert_eq!(out.cert.input("model_id"), Some(&b"stories15M-llama2c"[..]));
    assert_eq!(out.cert.input("weights_hash"), Some(&weights_hash[..]));
    assert!(out.cert.input("arch_hash").is_some());
    assert!(out.cert.input("prompt_tokens").is_some());
    let n_bytes = out.cert.input("n_new_tokens").unwrap();
    assert_eq!(u32::from_le_bytes(n_bytes.try_into().unwrap()), 2);

    // Outputs:
    let tokens_bytes = out.cert.output("output_tokens").unwrap();
    assert_eq!(tokens_bytes.len(), out.generated_tokens.len() * 4);
    let tokens_hash = out.cert.output("output_tokens_hash").unwrap();
    assert_eq!(tokens_hash.len(), 32);
    assert_eq!(
        tokens_hash,
        blake3::hash(tokens_bytes).as_bytes().as_slice(),
        "output_tokens_hash must match BLAKE3(output_tokens)"
    );
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn stories15m_real_weights_cert_and_replay() {
    let path = asset_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let blob = std::fs::read(&path).expect("read stories15M.bin");
    let cfg = Config::from_header(&blob).expect("parse config");
    let weights = Weights::from_blob(&blob, &cfg).expect("parse weights");
    let weights_hash = *blake3::hash(&blob).as_bytes();

    let req = inference::Request {
        model_id: "stories15M-llama2c",
        prompt_tokens: &[1],
        n_new_tokens: 5,
    };

    // First run — "the prover."
    let first = inference::run(&cfg, &weights, &weights_hash, &req).expect("first run");
    eprintln!("generated: {:?}", first.generated_tokens);
    eprintln!("cert digest: {}", first.cert.digest_hex());
    // 5 NEW tokens after the prompt [BOS=1]. Greedy decoding under
    // the reference implementation semantics (prompt populates KV, predictions
    // are tokens AT positions >= prompt_len). This sequence is the
    // model's deterministic continuation: 9038 ("▁Once") was the
    // prediction AT pos=0 (used as input at pos=1, not pushed); from
    // there each subsequent argmax becomes the next "Once upon a
    // time," then the next narrative token.
    assert_eq!(first.generated_tokens, vec![2501, 263, 931, 29892, 727]);

    // Second run from scratch — "the verifier." Fresh state, same
    // inputs, must produce bit-identical cert.
    let blob_again = std::fs::read(&path).expect("read again");
    assert_eq!(*blake3::hash(&blob_again).as_bytes(), weights_hash);
    let weights_again = Weights::from_blob(&blob_again, &cfg).unwrap();
    let second = inference::run(&cfg, &weights_again, &weights_hash, &req).expect("second run");

    assert_eq!(
        first.cert.digest, second.cert.digest,
        "verifier's cert digest differs from prover's — verifiability broken"
    );
    assert_eq!(first.generated_tokens, second.generated_tokens);

    // Negative case: a tampered "verifier" who lies about the weights
    // hash (claims they used different weights) MUST produce a
    // different cert.
    let tampered_hash = [0u8; 32];
    let tampered = inference::run(&cfg, &weights_again, &tampered_hash, &req).expect("tampered");
    assert_ne!(
        first.cert.digest, tampered.cert.digest,
        "cert binding broken — tampered weights_hash should change digest"
    );
}
