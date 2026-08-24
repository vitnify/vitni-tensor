#![no_main]
//! Fuzz the GGUF parser. `GgufFile::parse` consumes UNTRUSTED file bytes (a model a
//! user downloaded, or one an attacker crafted), so it must be totally safe on any
//! input: never panic, never read out of bounds, never overflow -- only return `Err`
//! on malformed data. This is the highest-value unaudited surface in the engine: an
//! untrusted parser in a `no_std` crate.
use libfuzzer_sys::fuzz_target;
use vitni_tensor::model::gguf::GgufFile;

fuzz_target!(|data: &[u8]| {
    // A malformed GGUF must fail gracefully, not crash.
    let _ = GgufFile::parse(data);
});
