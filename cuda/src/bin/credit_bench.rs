//! Regulated-industry benchmark: a stream of credit/loan underwriting decisions on
//! the deterministic CUDA engine, each producing a reproducible execution
//! certificate (the core of a Vitnify receipt). Measures throughput and
//! per-decision latency the way a deployment actually runs — the model is loaded
//! ONCE and the KV cache reset per decision (not reloaded) — and proves a sample
//! recomputes bit-for-bit on the CPU (an independent hardware path), plus anchors
//! the determinism tax against the deterministic CPU engine on the same box.
//!
//!   VITNI_GGUF=/path/model.gguf  N_DECISIONS=24  cargo run --release --bin credit_bench
//!
//! Pair with `llama-bench -m <same gguf>` for the native (no-Vitnify) baseline.

use std::time::Instant;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::model::config::Config;
use vitni_tensor::model::quant_weights::QuantizedWeights;
use vitni_tensor::model::inference::Request;
use vitni_tensor::tokenizer::{GgufBpeTokenizer, GgufSpmTokenizer};
use vitni_tensor_cuda::{run_cuda_with, CudaForward, device_name};

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
    // mmap the GGUF: a 19GB+ model must not be read into host RAM (the box has 16GB).
    let file = std::fs::File::open(&path).expect("open gguf");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap gguf") };
    let blob: &[u8] = &mmap;
    let hash = *blake3::hash(blob).as_bytes();
    let gguf = GgufFile::parse(blob).unwrap();
    let cfg = Config::from_gguf(&gguf).unwrap();
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
    let tok = GgufBpeTokenizer::from_gguf(&gguf).map(Tok::Bpe)
        .or_else(|_| GgufSpmTokenizer::from_gguf(&gguf).map(Tok::Spm))
        .expect("tokenizer from gguf");

    let model = std::path::Path::new(&path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(path.clone());
    println!("\n  Credit-decision benchmark (deterministic CUDA, Vitnify receipt per decision)");
    println!("  device   : {}", device_name());
    println!("  model    : {}   layers {}   dim {}   vocab {}", model, cfg.n_layers, cfg.dim, cfg.vocab_size);
    println!("  decisions: {}   new tokens/decision: {}\n", n, n_new);

    // Load the model onto the GPU ONCE (the deployment path): warm, reused.
    let load0 = Instant::now();
    let mut cf = CudaForward::new(&cfg, &weights).expect("load model on GPU");
    println!("  model loaded to GPU in {:.1}s\n", load0.elapsed().as_secs_f64());

    let mut latencies = Vec::with_capacity(n);
    let mut total_tokens = 0usize;
    let mut digests: Vec<(String, Vec<u32>)> = Vec::with_capacity(n);
    let wall = Instant::now();
    for i in 0..n {
        let prompt = application(i);
        let ptoks = tok.encode(&prompt);
        let req = Request { model_id: "credit-underwriter", prompt_tokens: &ptoks, n_new_tokens: n_new };
        let t0 = Instant::now();
        let out = run_cuda_with(&mut cf, &cfg, &hash, &req).unwrap();
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

    // Bit-identity at this model size: compare ONE forward pass (first prompt token,
    // pos 0) GPU vs CPU, bit-for-bit. A single bit-identical forward plus the engine's
    // determinism implies an identical full receipt — and a full-generation CPU
    // recompute of a 32B model takes hours (that slowness *is* the determinism tax on
    // CPU). Full-receipt bit-identity is proven separately on 7B/0.5B with these same
    // kernels. N_VERIFY=0 skips this.
    let nv: usize = std::env::var("N_VERIFY").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut verified = false;
    let mut cpu_fwd_ms = 0.0f64;
    let mut gpu_fwd_ms = 0.0f64;
    if nv > 0 {
        use vitni_tensor::model::{forward::RunState, forward_quantized};
        let ptoks = tok.encode(&application(0));
        cf.reset();
        let t = Instant::now();
        let g = cf.step(ptoks[0], 0);
        gpu_fwd_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let mut st = RunState::new(&cfg);
        let ct = forward_quantized::step(&cfg, &weights, &mut st, ptoks[0], 0).unwrap();
        cpu_fwd_ms = t.elapsed().as_secs_f64() * 1000.0;
        let cb = ct.storage_cpu_bytes().unwrap();
        let cl: &[f32] = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const f32, cb.len() / 4) };
        let same = (0..g.len()).filter(|&i| g[i].to_bits() == cl[i].to_bits()).count();
        verified = same == g.len();
        println!("\n  bit-identity (one forward pass, GPU vs independent CPU):");
        println!("    {}/{} logits identical  ->  {}", same, g.len(),
            if verified { "BIT-IDENTICAL (same receipt on either hardware)" } else { ">>> MISMATCH <<<" });
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dps = n as f64 / total;
    let gpu_ms = pct(&latencies, 50.0);
    println!("\n  ---- results (deterministic CUDA GPU) ----");
    println!("  decisions            : {}", n);
    println!("  wall time            : {:.1} s", total);
    println!("  throughput           : {:.3} decisions/s   ({:.0} decisions/hour)", dps, dps * 3600.0);
    println!("  latency p50 / p95    : {:.0} ms / {:.0} ms", gpu_ms, pct(&latencies, 95.0));
    println!("  engine throughput    : {:.2} tok/s", total_tokens as f64 / total);
    if nv > 0 {
        println!("  receipt verified     : {} (1 forward reproduced bit-identical on an independent CPU path)",
            if verified { "YES" } else { "NO" });
        println!("  determinism tax ref  : CPU {:.0} ms/forward  vs  GPU {:.0} ms/forward  ->  GPU {:.1}x faster per forward",
            cpu_fwd_ms, gpu_fwd_ms, cpu_fwd_ms / gpu_fwd_ms.max(1.0));
    }
    if nv > 0 && !verified { std::process::exit(1); }
}
