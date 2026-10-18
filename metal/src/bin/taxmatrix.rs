//! Consolidated determinism-tax benchmark.
//!
//! Runs the SAME model through the deterministic CPU path and the deterministic GPU
//! (Metal) path, times per-token decode, and asserts the two land on a BIT-IDENTICAL
//! certificate so the comparison is apples-to-apples. Build default for serial CPU,
//! `--features parallel` for the std-parallel CPU path. Pair it with llama.cpp
//! (`llama-bench`) for the native baseline row; `tax-matrix.sh` drives the whole matrix.
//!
//!   VITNI_GGUF=/path/model.gguf  N_TOK=32  cargo run --release --bin taxmatrix
//!   VITNI_GGUF=/path/model.gguf           cargo run --release --features parallel --bin taxmatrix

use std::time::Instant;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::model::config::Config;
use vitni_tensor::model::quant_weights::QuantizedWeights;
use vitni_tensor::model::forward::RunState;
use vitni_tensor::model::forward_quantized;
use vitni_tensor::model::inference::{self, Request};
use vitni_tensor_metal::{run_metal, MetalForward};

fn main() {
    let path = std::env::var("VITNI_GGUF").expect("set VITNI_GGUF to a GGUF model");
    let n: usize = std::env::var("N_TOK").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let blob = std::fs::read(&path).expect("read gguf");
    let hash = *blake3::hash(&blob).as_bytes();
    let gguf = GgufFile::parse(&blob).unwrap();
    let cfg = Config::from_gguf(&gguf).unwrap();
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();

    let cpu_label = if cfg!(feature = "parallel") { "deterministic CPU (parallel)" } else { "deterministic CPU (serial)" };
    let model = std::path::Path::new(&path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(path.clone());
    println!("\n  determinism-tax matrix  —  model: {}  ({} decode steps, token-at-a-time)\n", model, n);

    // ---- per-token decode timing, with warmup ----
    let tok = 1u32;
    let mut st = RunState::new(&cfg);
    for pos in 0..4 { forward_quantized::step(&cfg, &weights, &mut st, tok, pos).unwrap(); }
    let t = Instant::now();
    for pos in 4..4 + n { let _ = forward_quantized::step(&cfg, &weights, &mut st, tok, pos).unwrap(); }
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

    let mut mf = MetalForward::new(&cfg, &weights).unwrap();
    for pos in 0..4 { mf.step(tok, pos); }
    let t = Instant::now();
    for pos in 4..4 + n { let _ = mf.step(tok, pos); }
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // ---- bit-identity: full generation, compare certificate digests ----
    let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
    let req = Request { model_id: "taxmatrix", prompt_tokens: &prompt, n_new_tokens: 8 };
    let cpu_cert = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
    let gpu_cert = run_metal(&cfg, &weights, &hash, &req).unwrap();
    let bitident = cpu_cert.cert.digest == gpu_cert.cert.digest
        && cpu_cert.generated_tokens == gpu_cert.generated_tokens;

    // ---- report ----
    println!("  {:<30}{:>10}{:>11}{:>9}", "backend", "ms/tok", "tok/s", "vs CPU");
    println!("  {}", "-".repeat(60));
    let row = |name: &str, ms: f64| {
        println!("  {:<30}{:>10.1}{:>11.1}{:>8.1}x", name, ms, 1000.0 / ms, cpu_ms / ms);
    };
    row(cpu_label, cpu_ms);
    row("deterministic GPU (Metal)", gpu_ms);
    println!("  {}", "-".repeat(60));

    let mut hex = String::new();
    for x in cpu_cert.cert.digest.iter().take(8) { hex.push_str(&format!("{:02x}", x)); }
    println!("\n  bit-identical  CPU == GPU : {}", if bitident { "YES" } else { ">>> NO <<<" });
    println!("  shared certificate digest : {}...\n", hex);
    if !bitident { std::process::exit(1); }
}
