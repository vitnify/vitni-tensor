//! vitni-receipt-metal — emit a `vitnify-receipt v2` model-computation digest for a run.
//!
//! Given a GGUF model and a prompt (comma-separated token IDs), run the deterministic
//! engine and print, as JSON, the generated tokens and the model-computation digest.
//! The vitnify SDK calls this and records the digest into the run's `llm_call` event,
//! so the signed execution receipt binds what the model actually computed.
//!
//! Usage: vitni-receipt-metal --gguf PATH --prompt "1,9038,2501" [--n 16]
extern crate alloc;
use vitni_tensor::model::{
    config::Config, gguf::GgufFile, inference, quant_weights::QuantizedWeights,
};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn arg(flag: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

fn main() {
    // Explicit preset for the retained optimizations; default backend behavior
    // remains unchanged. This is inherited by the SDK's subprocess bridge.
    let optimized = std::env::args().any(|a| a == "--optimized")
        || std::env::var("VITNI_METAL_OPTIMIZED").as_deref() == Ok("1");
    if optimized {
        for (key, value) in [
            ("VITNI_METAL_Q4K_VARIANT", "metadata"),
            ("VITNI_METAL_RMS_COOPERATIVE", "1"),
            ("VITNI_METAL_RMS_VARIANT", "no_contract"),
            ("VITNI_METAL_Q6K_COOPERATIVE", "1"),
            ("VITNI_METAL_ATTENTION_COOPERATIVE", "1"),
            ("VITNI_METAL_CONCURRENT_DISPATCH", "1"),
        ] {
            std::env::set_var(key, value);
        }
    }
    let path = arg("--gguf").expect("usage: --gguf PATH --prompt \"1,2,3\" [--n 16]");
    let prompt: Vec<u32> = arg("--prompt")
        .expect("--prompt \"comma,separated,token,ids\"")
        .split(',')
        .map(|s| s.trim().parse().expect("token ids must be integers"))
        .collect();
    let n: usize = arg("--n")
        .unwrap_or_else(|| "16".to_string())
        .parse()
        .expect("--n integer");

    let blob = std::fs::read(&path).expect("read gguf");
    let weights_hash = *blake3::hash(&blob).as_bytes();
    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let cfg = Config::from_gguf(&gguf).expect("config from gguf");
    // The current Metal attention implementation has a fixed 512-score tile,
    // and RMS uses one canonical chunk. Reject unsupported extents before GPU
    // encoding while leaving the configuration bound by the receipt unchanged.
    assert!(
        cfg.dim <= 8192 && cfg.head_size() <= 8192 && cfg.hidden_dim <= 64 * 8192,
        "model dimensions exceed this Metal CLI's supported bounds"
    );
    assert!(
        prompt
            .len()
            .checked_add(n)
            .is_some_and(|len| len <= cfg.seq_len.min(512)),
        "prompt plus generated tokens exceeds the Metal CLI context bound"
    );
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).expect("weights from gguf");
    use vitni_tensor::model::gguf::GgufTensorType;
    let supported = |t: &vitni_tensor::model::quant_weights::QuantTensor, k: usize| match t.dtype {
        GgufTensorType::Q4_K | GgufTensorType::Q6_K => k % 256 == 0,
        GgufTensorType::Q5_0 | GgufTensorType::Q8_0 => k % 32 == 0,
        _ => false,
    };
    for layer in &weights.layers {
        for tensor in [
            &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w1, &layer.w3,
        ] {
            assert!(
                supported(tensor, cfg.dim),
                "unsupported Metal projection dtype or block width"
            );
        }
        assert!(
            supported(&layer.w2, cfg.hidden_dim),
            "unsupported Metal down-projection dtype or block width"
        );
    }
    assert!(
        supported(
            weights
                .wcls
                .as_ref()
                .unwrap_or(&weights.token_embedding_table),
            cfg.dim
        ),
        "unsupported Metal output dtype or block width"
    );

    let model_id = arg("--model-id").unwrap_or_else(|| "vitni-receipt".to_string());
    let req = inference::Request {
        model_id: &model_id,
        prompt_tokens: &prompt,
        n_new_tokens: n,
    };
    let a = vitni_tensor_metal::run_metal(&cfg, &weights, &weights_hash, &req).expect("run");

    // The shipped `model_digest` is v2 (binds the numerical REGIME). The frozen v1
    // digest for the same run is emitted too, so a pre-regime (v1) receipt stays
    // reproducible and the two are never confused.
    let c = &a.cert;
    let digest_v1 = vitni_tensor::cert::builder::compute_digest_v1(
        &c.inputs,
        &c.outputs,
        &c.ops,
        &c.activations,
        &c.interventions,
    );

    let toks: Vec<String> = a.generated_tokens.iter().map(|t| t.to_string()).collect();
    println!(
        "{{\"model_digest\":\"{}\",\"regime\":\"{}\",\"model_digest_v1\":\"{}\",\"weights_hash\":\"{}\",\"tokens\":[{}]}}",
        hex(&c.digest),
        vitni_tensor::cert::builder::REGIME,
        hex(&digest_v1),
        hex(&weights_hash),
        toks.join(",")
    );
}
