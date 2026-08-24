#![no_main]
//! Fuzz the k-quant dequantizers. Each takes UNTRUSTED quantized bytes (the tensor
//! payload of a GGUF an attacker crafted) and must never panic / read out of bounds /
//! overflow on a malformed or truncated block -- only return Err. Covers every shipped
//! quant type (Q4_0/Q5_0/Q8_0/Q4_K/Q6_K).
use libfuzzer_sys::fuzz_target;
use vitni_tensor::ops::quant;

fuzz_target!(|data: &[u8]| {
    let _ = quant::dequantize_q4_0(data);
    let _ = quant::dequantize_q5_0(data);
    let _ = quant::dequantize_q8_0(data);
    let _ = quant::dequantize_q4_k(data);
    let _ = quant::dequantize_q6_k(data);
});
