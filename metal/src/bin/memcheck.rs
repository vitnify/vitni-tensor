//! Memory-leak probe for the Metal decode loop. Calls `MetalForward::step` many times
//! (fixed position -- the leak, if any, is per CALL, not per position) and samples this
//! process's RSS. Flat RSS = no leak; linear growth = autoreleased Metal objects piling up.
//!
//!   VITNI_GGUF=/path/tiny.gguf  STEPS=3000  cargo run --release --bin memcheck

use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::model::config::Config;
use vitni_tensor::model::quant_weights::QuantizedWeights;
use vitni_tensor_metal::MetalForward;

fn rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &pid]).output();
    out.ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn main() {
    let path = std::env::var("VITNI_GGUF").expect("set VITNI_GGUF");
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let blob = std::fs::read(&path).unwrap();
    let gguf = GgufFile::parse(&blob).unwrap();
    let cfg = Config::from_gguf(&gguf).unwrap();
    let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
    let mut mf = MetalForward::new(&cfg, &weights).unwrap();

    for _ in 0..8 { mf.step(1, 5); } // warmup, let buffers settle
    let base = rss_mb();
    println!("steps      RSS(MB)   delta");
    for i in 1..=steps {
        let _ = mf.step(1, 5); // fixed pos: isolate per-call allocation
        if i % 500 == 0 {
            let r = rss_mb();
            println!("{:>6}   {:>8.1}   {:>+7.1}", i, r, r - base);
        }
    }
}
