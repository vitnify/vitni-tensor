//! gen-anchors — regenerate every published determinism anchor in ONE command.
//!
//! The certificate digests, the pinned reduction hashes, and the cross-vendor
//! model digests are all functions of the numerical REGIME. When the regime
//! changes they all change, and hand-syncing them across tests, the spec, and
//! the paper is the "heavy lift". This binary is the single source of truth:
//! it runs the deterministic computations and writes `regime-manifest.json`.
//! Tests, the spec, and the paper reference that manifest; regenerating is
//!
//!   cargo run --release --bin gen-anchors > regime-manifest.json
//!
//! Model paths default to the local checkouts; override any with the matching
//! VITNI_GGUF_* env var. Missing models are skipped (recorded as null).
extern crate alloc;
use vitni_tensor::cert::builder::{compute_digest_v1, REGIME};
use vitni_tensor::model::{config::Config, gguf::GgufFile, inference, quant_weights::QuantizedWeights};
use vitni_tensor::{Shape, Storage, Tensor};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// FNV-1a over an f32 slice's little-endian bit patterns (the pin hash form).
fn fnv1a(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Reproduce `matmul_reduction_bits_are_pinned`: FNV over the (4,64,4) LCG matmul.
fn matmul_pin() -> u64 {
    let (m, k, n) = (4usize, 64usize, 4usize);
    let mut s: u64 = 0x1234;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
    let ta = Tensor::from_f32(a, Shape::new(&[m, k]).unwrap()).unwrap();
    let tb = Tensor::from_f32(b, Shape::new(&[k, n]).unwrap()).unwrap();
    let out = ta.matmul(&tb).unwrap();
    let Storage::Cpu(os) = out.storage() else { panic!("cpu") };
    fnv1a(os.as_f32_slice())
}

/// (model_digest_v2, model_digest_v1, generated_tokens) for one model, or None.
/// The digest binds `model_id`, so the canonical model name is used as the id —
/// the anchor is then self-describing and reproducible from the manifest alone.
fn model_digest(name: &str, path: &str, prompt: &[u32], n: usize) -> Option<(String, String, Vec<u32>)> {
    let blob = std::fs::read(path).ok()?;
    let wh = *blake3::hash(&blob).as_bytes();
    let gguf = GgufFile::parse(&blob).ok()?;
    let cfg = Config::from_gguf(&gguf).ok()?;
    let w = QuantizedWeights::from_gguf(&gguf, &cfg).ok()?;
    let req = inference::Request { model_id: name, prompt_tokens: prompt, n_new_tokens: n };
    let a = inference::run_quantized(&cfg, &w, &wh, &req).ok()?;
    let c = &a.cert;
    let v1 = compute_digest_v1(&c.inputs, &c.outputs, &c.ops, &c.activations, &c.interventions);
    Some((hex(&c.digest), hex(&v1), a.generated_tokens))
}

fn main() {
    // Standard reproduction vector: "Once upon a time," (TinyLlama tokenization;
    // all IDs < 32000 so it is a valid, well-defined prompt for every model).
    let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
    let n_new = 20usize;
    let models: &[(&str, &str, &str)] = &[
        ("tinyllama-1.1b-Q4_K_M", "VITNI_GGUF_TINYLLAMA", "/Users/nickp/Downloads/vitnify_test/tinyllama-Q4_K_M.gguf"),
        ("mistral-7b-Q4_K_M", "VITNI_GGUF_MISTRAL", "/Users/nickp/models/mistral-7b-v0.1.Q4_K_M.gguf"),
        ("qwen2.5-0.5b-Q4_K_M", "VITNI_GGUF_QWEN", "/Users/nickp/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ];

    let mut model_lines: Vec<String> = Vec::new();
    for (name, env, default) in models {
        let path = std::env::var(env).unwrap_or_else(|_| default.to_string());
        match model_digest(name, &path, &prompt, n_new) {
            Some((v2, v1, toks)) => {
                let t: Vec<String> = toks.iter().map(|x| x.to_string()).collect();
                model_lines.push(format!(
                    "    \"{}\": {{ \"model_digest\": \"{}\", \"model_digest_v1\": \"{}\", \"tokens\": [{}] }}",
                    name, v2, v1, t.join(", ")
                ));
            }
            None => model_lines.push(format!("    \"{}\": null", name)),
        }
    }

    let prompt_s: Vec<String> = prompt.iter().map(|x| x.to_string()).collect();
    println!("{{");
    println!("  \"regime\": \"{}\",", REGIME);
    println!("  \"prompt_tokens\": [{}],", prompt_s.join(", "));
    println!("  \"n_new_tokens\": {},", n_new);
    println!("  \"pins\": {{");
    println!("    \"matmul_reduction\": \"{:#018x}\"", matmul_pin());
    println!("  }},");
    println!("  \"models\": {{");
    println!("{}", model_lines.join(",\n"));
    println!("  }}");
    println!("}}");
}
