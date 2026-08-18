//! Dump the engine's per-step logits for a prompt, for numerical validation against
//! a deterministic CPU reference (llama.cpp). Writes a binary file: [vocab:u32][n_new:u32]
//! then n_new rows of vocab f32 logits (LE bit patterns), one per generated-token decision.
//! Env: VITNI_GGUF, LOGIT_OUT (path), PROMPT_TOKS ("1,9038,.."), N_NEW (default 16).
extern crate alloc;
use std::io::Write;
use vitni_tensor::model::{config::Config, forward::RunState, forward_quantized, gguf::GgufFile,
    quant_weights::QuantizedWeights};

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn logit_dump() {
    let path = std::env::var("VITNI_GGUF").expect("VITNI_GGUF");
    let out = std::env::var("LOGIT_OUT").expect("LOGIT_OUT");
    let prompt: Vec<u32> = std::env::var("PROMPT_TOKS").expect("PROMPT_TOKS")
        .split(',').map(|s| s.trim().parse().unwrap()).collect();
    let n_new: usize = std::env::var("N_NEW").ok().and_then(|s| s.parse().ok()).unwrap_or(16);

    let blob = std::fs::read(&path).unwrap();
    let gguf = GgufFile::parse(&blob).unwrap();
    let mut cfg = Config::from_gguf(&gguf).unwrap();
    cfg.seq_len = 1024;
    let w = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();

    let mut state = RunState::new(&cfg);
    let prompt_len = prompt.len();
    let total = prompt_len + n_new;
    let mut cur = prompt[0];
    let mut gen: Vec<u32> = Vec::new();
    let mut file = std::fs::File::create(&out).unwrap();
    file.write_all(&(cfg.vocab_size as u32).to_le_bytes()).unwrap();
    file.write_all(&(n_new as u32).to_le_bytes()).unwrap();

    let mut written = 0usize;
    for pos in 0..total {
        let logits = forward_quantized::step(&cfg, &w, &mut state, cur, pos).unwrap();
        let bytes = logits.storage_cpu_bytes().unwrap(); // vocab * 4 bytes, f32 LE
        let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
        for (i, c) in bytes.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v > bv { bv = v; bi = i; }
        }
        let argmax = bi as u32;
        if pos + 1 >= prompt_len && written < n_new {
            file.write_all(bytes).unwrap();
            written += 1;
        }
        let next = if pos + 1 < prompt_len { prompt[pos + 1] } else { argmax };
        if pos + 1 >= prompt_len { gen.push(next); }
        cur = next;
        if written >= n_new { break; }
    }
    eprintln!("LOGIT_DUMP tokens={:?} vocab={} steps={}", gen, cfg.vocab_size, written);
}
