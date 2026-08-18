//! Qwen2.5-0.5B-Instruct (Q6_K/Q8_0 GGUF) end-to-end through the deterministic
//! engine. This is the first STRUCTURALLY DIFFERENT architecture: Qwen2 has
//! QKV biases and NeoX-style (split-half) RoPE, unlike every Llama-family model
//! (TinyLlama, Mistral). It also exercises the Q8_0 matmul/embedding path.
//! rope_theta (1e6) and rms_eps (1e-6) are read from the GGUF, not hard-coded.
extern crate alloc;
use vitni_tensor::model::{config::Config, gguf::GgufFile, inference, quant_weights::QuantizedWeights};

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn qwen2_5_gguf_generate_certify_replay() {
    let path = std::env::var("QWEN_GGUF").expect("set QWEN_GGUF to the qwen2.5 gguf path");
    let blob = std::fs::read(&path).expect("read qwen gguf");
    eprintln!("loaded gguf: {} MB", blob.len() / (1024 * 1024));
    let weights_hash = *blake3::hash(&blob).as_bytes();

    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let mut cfg = Config::from_gguf(&gguf).expect("config from gguf");
    cfg.seq_len = 512; // cap KV cache (native context 32768)
    eprintln!("config: dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq(capped)={}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len);
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).expect("weights from gguf");
    eprintln!("arch params: rope_theta={} rms_eps={} rope_neox={}",
        weights.rope_theta, weights.rms_eps, weights.rope_neox);

    // "Once upon a time," tokenized by Qwen2.5's own byte-level BPE (matches llama.cpp).
    let prompt: Vec<u32> = vec![12522, 5193, 264, 882, 11];
    let req = inference::Request {
        model_id: "qwen2.5-0.5b-instruct",
        prompt_tokens: &prompt,
        n_new_tokens: 16,
    };

    let a = inference::run_quantized(&cfg, &weights, &weights_hash, &req).expect("run a");
    let b = inference::run_quantized(&cfg, &weights, &weights_hash, &req).expect("run b (replay)");

    eprintln!("generated tokens: {:?}", a.generated_tokens);
    eprintln!("weights hash   : {}", hex(&weights_hash));
    eprintln!("cert digest    : {}", hex(&a.cert.digest));
    assert_eq!(a.generated_tokens, b.generated_tokens, "Qwen generation non-deterministic");
    assert_eq!(a.cert.digest, b.cert.digest, "Qwen cert digest did not reproduce");
    eprintln!("OK — Qwen2.5 (QKV bias + NeoX RoPE + Q8_0) generated, certified, reproduced bit-identically");
}
