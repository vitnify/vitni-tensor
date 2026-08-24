#![no_main]
//! Fuzz parse + Config derivation. A malicious GGUF that PARSES can still carry hostile
//! metadata (dims, head counts, rope params). Deriving a Config -- divisions like
//! head_size = dim / n_heads, casts, derived sizes -- must not panic (divide-by-zero,
//! overflow) or over-allocate on untrusted values. Only Err.
use libfuzzer_sys::fuzz_target;
use vitni_tensor::model::gguf::GgufFile;
use vitni_tensor::model::config::Config;

fuzz_target!(|data: &[u8]| {
    if let Ok(gguf) = GgufFile::parse(data) {
        let _ = Config::from_gguf(&gguf);
    }
});
