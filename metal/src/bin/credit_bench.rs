//! Regulated-industry benchmark: a stream of credit/loan decisions on the deterministic
//! GPU engine, each producing a reproducible execution certificate (the core of a Vitnify
//! receipt). Measures throughput, per-decision latency, and proves a sample recomputes
//! bit-for-bit on the CPU (independent hardware path). Pair with llama-bench for the native
//! (no-Vitnify) baseline; credit-bench.sh drives the full comparison.
//!
//!   VITNI_GGUF=/path/model.gguf  N_DECISIONS=24  cargo run --release --bin credit_bench

use std::time::Instant;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::model::config::Config;
use vitni_tensor::model::quant_weights::QuantizedWeights;
use vitni_tensor::model::inference::{self, Request};
use vitni_tensor::tokenizer::{GgufBpeTokenizer, GgufSpmTokenizer};
use vitni_tensor_metal::run_metal;

enum Tok { Bpe(GgufBpeTokenizer), Spm(GgufSpmTokenizer) }
impl Tok {
    fn encode(&self, s: &str) -> Vec<u32> {
        match self {
            Tok::Bpe(t) => t.encode(s),
            Tok::Spm(t) => t.encode(s, true),
        }
    }
}

/// Deterministic, reproducible synthetic loan applications (same seed -> same stream).
fn application(i: usize) -> String {
    let age = 25 + (i * 7) % 40;
    let income = 35_000 + (i * 9_300) % 180_000;
    let score = 580 + (i * 23) % 260;
    let loan = 8_000 + (i * 4_700) % 60_000;
    let dti = 8 + (i * 5) % 48;
    let years = 1 + (i * 3) % 18;
    format!(
        "You are a loan underwriter. Decide APPROVE or DENY and give one short reason.\n\
         Applicant: age {age}, annual income ${income}, credit score {score}, \
         requested loan ${loan}, debt-to-income {dti}%, employed {years} years.\n\
         Decision:"
    )
}

fn pct(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() { return 0.0; }
    let idx = ((p / 100.0) * (sorted_ms.len() as f64 - 1.0)).round() as usize;
    sorted_ms[idx]
}

fn main() {
    let path = std::env::var("VITNI_GGUF").expect("set VITNI_GGUF");
    let n: usize = std::env::var("N_DECISIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let n_new = 32usize; // APPROVE/DENY + a short reason
    let blob = std::fs::read(&path).expect("read gguf");
    let hash = *blake3::hash(&blob).as_bytes();
    let gguf = GgufFile::parse(&blob).unwrap();
    let cfg = Config::from_gguf(&gguf).unwrap();
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
    let tok = GgufBpeTokenizer::from_gguf(&gguf).map(Tok::Bpe)
        .or_else(|_| GgufSpmTokenizer::from_gguf(&gguf).map(Tok::Spm))
        .expect("tokenizer from gguf");

    let model = std::path::Path::new(&path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(path.clone());
    println!("\n  Credit-decision benchmark (deterministic GPU, Vitnify receipt per decision)");
    println!("  model: {}   decisions: {}   new tokens/decision: {}\n", model, n, n_new);

    let mut latencies = Vec::with_capacity(n);
    let mut total_tokens = 0usize;
    let mut digests: Vec<(String, Vec<u32>)> = Vec::with_capacity(n);
    let wall = Instant::now();
    for i in 0..n {
        let prompt = application(i);
        let ptoks = tok.encode(&prompt);
        let req = Request { model_id: "credit-underwriter", prompt_tokens: &ptoks, n_new_tokens: n_new };
        let t0 = Instant::now();
        let out = run_metal(&cfg, &weights, &hash, &req).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        latencies.push(dt * 1000.0);
        total_tokens += ptoks.len() + n_new;
        let mut hex = String::new();
        for b in out.cert.digest.iter().take(8) { hex.push_str(&format!("{:02x}", b)); }
        digests.push((hex, out.generated_tokens.clone()));
        if i < 3 || i == n - 1 {
            println!("  decision {:>3}: {} tok prompt + {} new  ->  receipt {}...  ({:.2}s)",
                i + 1, ptoks.len(), n_new, digests[i].0, dt);
        }
    }
    let total = wall.elapsed().as_secs_f64();

    // bit-identity: recompute a few decisions on the CPU engine, assert the receipt matches.
    println!("\n  independent reproduction (recompute on CPU, no GPU): ");
    let mut repro_ok = 0;
    for &i in &[0usize, n / 2, n - 1] {
        let prompt = application(i);
        let ptoks = tok.encode(&prompt);
        let req = Request { model_id: "credit-underwriter", prompt_tokens: &ptoks, n_new_tokens: n_new };
        let cpu = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
        let mut hex = String::new();
        for b in cpu.cert.digest.iter().take(8) { hex.push_str(&format!("{:02x}", b)); }
        let ok = hex == digests[i].0 && cpu.generated_tokens == digests[i].1;
        if ok { repro_ok += 1; }
        println!("    decision {:>3}: GPU receipt {}  ==  CPU recompute {}  -> {}",
            i + 1, digests[i].0, hex, if ok { "MATCH" } else { ">>> MISMATCH <<<" });
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dps = n as f64 / total;
    println!("\n  ---- results (deterministic GPU) ----");
    println!("  decisions            : {}", n);
    println!("  wall time            : {:.1} s", total);
    println!("  throughput           : {:.2} decisions/s   ({:.0} decisions/hour)", dps, dps * 3600.0);
    println!("  latency p50 / p95    : {:.0} ms / {:.0} ms", pct(&latencies, 50.0), pct(&latencies, 95.0));
    println!("  engine throughput    : {:.2} tok/s", total_tokens as f64 / total);
    println!("  receipts reproduced  : {}/3 bit-identical on an independent CPU path", repro_ok);
    if repro_ok != 3 { std::process::exit(1); }
}
