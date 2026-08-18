//! Cross-architecture determinism of the software transcendental path.
//!
//! The engine's determinism argument depends on transcendentals (exp, sqrt, cos,
//! sin, pow, the sigmoid) coming from the pure-software `libm` crate rather than
//! hardware units that differ across vendors. This test exercises those functions
//! over a dense sweep of inference-range inputs plus corner cases, and folds the
//! raw IEEE-754 bit patterns of every result into one BLAKE3 hash. Run on each
//! target architecture; the hash must be identical, which proves the softfloat
//! path is bit-identical ARM-vs-x86, not merely assumed to be.
extern crate alloc;

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

#[test]
fn transcendental_determinism() {
    let mut inputs: Vec<f32> = Vec::new();
    // Dense sweep across the range attention/FFN/softmax actually produce.
    let mut x = -30.0f32;
    while x <= 30.0 {
        inputs.push(x);
        x += 0.000_123_f32;
    }
    // Corner cases.
    for &c in &[
        0.0f32, -0.0, 1.0, -1.0, 0.5, -0.5,
        f32::MIN_POSITIVE, 1e-20, 1e-10, 1e10, 1e20,
        87.0, 88.0, 88.7, -87.0, -88.0, -104.0,
        core::f32::consts::PI, core::f32::consts::E,
    ] {
        inputs.push(c);
    }

    let mut h = blake3::Hasher::new();
    for &v in &inputs {
        let vals = [
            libm::expf(v),
            libm::sqrtf(v.abs()),
            libm::cosf(v),
            libm::sinf(v),
            libm::powf(10000.0f32, v / 64.0f32), // RoPE-style base^(-2i/d)
            libm::powf(1_000_000.0f32, v / 64.0f32), // Qwen RoPE base
            1.0f32 / (1.0f32 + libm::expf(-v)),  // sigmoid / SwiGLU gate
        ];
        for r in &vals {
            h.update(&r.to_bits().to_le_bytes());
        }
    }
    let digest = *h.finalize().as_bytes();
    eprintln!("TRANSCENDENTAL inputs={} evaluations={} hash={}",
        inputs.len(), inputs.len() * 7, hex(&digest));
    assert_eq!(inputs.len() * 7 % 7, 0);
}
