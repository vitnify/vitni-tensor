//! Large determinism sweep. Runs many diverse input token sequences across a
//! range of prompt lengths and generation lengths through the engine, and folds
//! every per-run certificate digest into ONE aggregate BLAKE3. Cross-host, the
//! aggregate must match — that one number certifies bit-identity over the whole
//! matrix. Inputs are generated deterministically (a fixed LCG), so every host
//! builds the identical matrix without any shared file.
//!
//! Env: VITNI_GGUF (model path), SWEEP_N (random prompts per length, default 25),
//!      SWEEP_MAXLEN (largest prompt length, default 128 — lower it on slow hosts).
extern crate alloc;
use vitni_tensor::model::{config::Config, gguf::GgufFile, inference, quant_weights::QuantizedWeights};

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

/// Deterministic LCG so every host generates the identical prompt set.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 16
    }
}

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn determinism_sweep() {
    let path = std::env::var("VITNI_GGUF").expect("set VITNI_GGUF");
    let n_rand: usize = std::env::var("SWEEP_N").ok().and_then(|s| s.parse().ok()).unwrap_or(25);
    let max_len: usize = std::env::var("SWEEP_MAXLEN").ok().and_then(|s| s.parse().ok()).unwrap_or(128);

    let blob = std::fs::read(&path).expect("read gguf");
    let wh = *blake3::hash(&blob).as_bytes();
    let gguf = GgufFile::parse(&blob).expect("parse gguf");
    let mut cfg = Config::from_gguf(&gguf).expect("config");
    cfg.seq_len = 512;
    let w = QuantizedWeights::from_gguf(&gguf, &cfg).expect("weights");
    let vocab = cfg.vocab_size as u64;

    // Diverse, deterministic prompt set: structured edge cases + seeded random,
    // over several lengths. Categories: repetitive, sequential, max-token-id,
    // and pseudo-random token streams.
    let lens: Vec<usize> = [4usize, 16, 64, 128].iter().copied().filter(|&l| l <= max_len).collect();
    let gens = [8usize, 32];
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut lcg = Lcg(0x9E3779B97F4A7C15);
    for &l in &lens {
        prompts.push(vec![1u32; l]);
        prompts.push((0..l).map(|i| (i as u32) % (vocab as u32)).collect());
        prompts.push(vec![(vocab - 1) as u32; l]);
        for _ in 0..n_rand {
            prompts.push((0..l).map(|_| (lcg.next() % vocab) as u32).collect());
        }
    }

    let mut agg = blake3::Hasher::new();
    let mut runs = 0u64;
    let mut gen_tokens = 0u64;
    for p in &prompts {
        for &g in &gens {
            if p.len() + g > cfg.seq_len { continue; }
            let req = inference::Request { model_id: "sweep", prompt_tokens: p, n_new_tokens: g };
            let out = inference::run_quantized(&cfg, &w, &wh, &req).expect("run");
            agg.update(&out.cert.digest);
            runs += 1;
            gen_tokens += out.generated_tokens.len() as u64;
        }
    }
    let a = *agg.finalize().as_bytes();
    eprintln!("SWEEP runs={} generated_tokens={} prompts={} layers={} vocab={}",
        runs, gen_tokens, prompts.len(), cfg.n_layers, cfg.vocab_size);
    eprintln!("weights hash   : {}", hex(&wh));
    eprintln!("aggregate hash : {}", hex(&a));
    assert!(runs > 0);
}
