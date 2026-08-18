//! Probe a Qwen2.5 GGUF: arch, rope/rms hparams, config dims, tensor-dtype
//! histogram, and the dtype/shape of the load-bearing tensors (embeddings,
//! lm_head, QKV biases). Read-only — informs the Qwen forward-pass work.
//! Run: QWEN_GGUF=/path/to/qwen.gguf cargo test --release --test qwen_probe -- --nocapture
extern crate alloc;
use std::collections::BTreeMap;
use vitni_tensor::model::{config::Config, gguf::GgufFile};

#[test]
#[ignore = "needs a model file; set the model path and run with --ignored"]
fn qwen_probe() {
    let path = std::env::var("QWEN_GGUF").expect("set QWEN_GGUF");
    let blob = std::fs::read(&path).expect("read qwen gguf");
    let gguf = GgufFile::parse(&blob).expect("parse gguf");

    eprintln!("arch           = {}", gguf.metadata_str("general.architecture").unwrap_or("?"));
    eprintln!("rope.freq_base = {:?}", gguf.metadata_value("qwen2.rope.freq_base"));
    eprintln!("rms_eps        = {:?}", gguf.metadata_value("qwen2.attention.layer_norm_rms_epsilon"));

    match Config::from_gguf(&gguf) {
        Ok(cfg) => eprintln!(
            "cfg: dim={} hidden={} layers={} heads={} kv_heads={} head_size={} vocab={} seq={} shared={}",
            cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_size(), cfg.vocab_size, cfg.seq_len, cfg.shared_weights),
        Err(e) => eprintln!("Config::from_gguf ERROR: {:?}", e),
    }

    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for t in &gguf.tensors {
        *hist.entry(format!("{:?}", t.dtype)).or_insert(0) += 1;
    }
    eprintln!("dtype histogram ({} tensors):", gguf.tensors.len());
    for (k, v) in &hist {
        eprintln!("  {:>4}  {}", v, k);
    }
    for n in [
        "token_embd.weight", "output.weight",
        "blk.0.attn_q.weight", "blk.0.attn_q.bias",
        "blk.0.attn_k.bias", "blk.0.attn_v.bias",
        "blk.0.ffn_down.weight", "blk.0.attn_norm.weight",
    ] {
        eprintln!("  {:<26} {:?}", n,
            gguf.tensor(n).map(|t| (format!("{:?}", t.dtype), t.shape.clone())));
    }
}
