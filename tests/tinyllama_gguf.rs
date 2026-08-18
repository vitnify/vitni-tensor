//! MILESTONE: run a REAL ~1B model (TinyLlama-1.1B-Chat, Q4_K_M GGUF) through the
//! deterministic engine end-to-end — load, generate, certify, and reproduce
//! bit-identically. Scales the engine from the 15M toy to a real quantized model,
//! reusing the existing GGUF loader + GQA + KV-cache + Q4_K/Q6_K kernels + cert harness.
//!
//! TinyLlama is Llama-2 architecture: rope_theta=10000 (matches the engine's hard-coded
//! value), no QKV bias, GQA (32 q-heads / 4 kv-heads) — so it needs no code changes,
//! only this driver. Prompt is tokenized offline by the HF TinyLlama tokenizer; the cert
//! binds token IDs, so no in-engine tokenizer is required for this milestone.
extern crate alloc;
use vitni_tensor::model::{config::Config, gguf::GgufFile, inference, quant_weights::QuantizedWeights};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn tinyllama_gguf_generate_certify_replay() {
    // VITNI_GGUF overrides the path (used for the cross-vendor runs)
    let path = std::env::var("VITNI_GGUF").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../userspace/the reference implementation/assets/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").to_string()
    });
    let blob = std::fs::read(&path).expect("read tinyllama gguf");
    eprintln!("loaded gguf: {} MB", blob.len() / (1024 * 1024));
    let weights_hash = *blake3::hash(&blob).as_bytes();

    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let cfg = Config::from_gguf(&gguf).expect("config from gguf");
    eprintln!(
        "config: dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq={}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len
    );
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).expect("weights from gguf");

    // "Once upon a time," tokenized by the TinyLlama HF tokenizer (same vocab as the GGUF)
    let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
    let req = inference::Request {
        model_id: "tinyllama-1.1b-chat-Q4_K_M",
        prompt_tokens: &prompt,
        n_new_tokens: 20,
    };

    let a = inference::run_quantized(&cfg, &weights, &weights_hash, &req).expect("run a");
    let b = inference::run_quantized(&cfg, &weights, &weights_hash, &req).expect("run b (replay)");

    eprintln!("generated tokens: {:?}", a.generated_tokens);
    eprintln!("weights hash   : {}", hex(&weights_hash));
    eprintln!("cert digest    : {}", hex(&a.cert.digest));

    assert_eq!(
        a.generated_tokens, b.generated_tokens,
        "REAL 1B model generation was NON-deterministic across runs"
    );
    assert_eq!(
        a.cert.digest, b.cert.digest,
        "REAL 1B model certificate digest did not reproduce"
    );
    eprintln!("OK — real 1B model generated, certified, and reproduced bit-identically");
}
