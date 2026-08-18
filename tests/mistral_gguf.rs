//! SCALE CHECK: run the 7B model (Mistral-7B-v0.1, Q4_K_M GGUF, 4.1 GB) through the
//! deterministic engine. Same driver pattern as TinyLlama; Mistral is GQA (32 q / 8 kv),
//! rope_theta=10000, no QKV bias -> no engine changes needed. Only difference: cap seq_len
//! so the KV cache (context_length=32768 -> ~8.6 GB) is allocatable.
extern crate alloc;
use vitni_tensor::model::{config::Config, gguf::GgufFile, inference, quant_weights::QuantizedWeights};

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn mistral7b_gguf_generate_certify_replay() {
    // VITNI_GGUF overrides the path (used for the cross-hardware runs)
    let path = std::env::var("VITNI_GGUF").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../userspace/the reference implementation/assets/mistral-7b-v0.1.Q4_K_M.gguf").to_string()
    });
    let blob = std::fs::read(&path).expect("read mistral gguf");
    eprintln!("loaded gguf: {} MB", blob.len() / (1024 * 1024));
    let weights_hash = *blake3::hash(&blob).as_bytes();

    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let mut cfg = Config::from_gguf(&gguf).expect("config from gguf");
    cfg.seq_len = 512;      // cap KV-cache alloc (native context is 32768 -> ~8.6 GB)
    eprintln!("config: dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq(capped)={}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len);
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).expect("weights from gguf");

    // "Once upon a time," tokenized by MISTRAL's OWN vocab (from the GGUF, via llama.cpp)
    let prompt: Vec<u32> = vec![1, 5713, 3714, 264, 727, 28725];
    let req = inference::Request { model_id: "mistral-7b-v0.1-Q4_K_M", prompt_tokens: &prompt, n_new_tokens: 12 };

    let a = inference::run_quantized(&cfg, &weights, &weights_hash, &req).expect("run a");
    let b = inference::run_quantized(&cfg, &weights, &weights_hash, &req).expect("run b (replay)");

    eprintln!("generated tokens: {:?}", a.generated_tokens);
    eprintln!("weights hash   : {}", hex(&weights_hash));
    eprintln!("cert digest    : {}", hex(&a.cert.digest));
    assert_eq!(a.generated_tokens, b.generated_tokens, "7B generation non-deterministic");
    assert_eq!(a.cert.digest, b.cert.digest, "7B cert digest did not reproduce");
    eprintln!("OK — 7B model generated, certified, and reproduced bit-identically");
}
