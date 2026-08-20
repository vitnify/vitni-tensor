//! Element-wise unary ops. phase 2: SiLU (Swish), needed for SwiGLU FFN.
//! Additional unaries (neg, sqrt, exp, gelu) added as model ports demand.

use crate::{
    error::{Error, Result},
    storage::Storage,
    tensor::Tensor,
    DType,
};
use alloc::vec::Vec;

/// SiLU activation: `silu(x) = x * sigmoid(x) = x / (1 + e^-x)`.
///
/// Used in SwiGLU FFN (Llama, Mistral, Phi family). Identical to
/// Candle's `silu`. Uses `libm::expf` for `no_std` exponentiation.
pub(crate) fn silu(t: &Tensor) -> Result<Tensor> {
    apply_f32(t, "silu", |x| x / (1.0 + libm::expf(-x)))
}

/// GELU exact: `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`. Used in
/// BERT, GPT-2, Phi-2. Approximations exist (tanh-based) but for
/// determinism we use the exact form.
pub(crate) fn gelu(t: &Tensor) -> Result<Tensor> {
    // SQRT_1_2 = 1/sqrt(2)
    const SQRT_1_2: f32 = 0.707_106_77;
    apply_f32(t, "gelu", |x| 0.5 * x * (1.0 + libm::erff(x * SQRT_1_2)))
}

/// Reciprocal: `1 / x`. Used in normalization layers.
#[allow(dead_code)]
pub(crate) fn recip(t: &Tensor) -> Result<Tensor> {
    apply_f32(t, "recip", |x| 1.0 / x)
}

/// In-place-style F32 unary map. Returns a fresh contiguous tensor.
fn apply_f32(t: &Tensor, op: &'static str, f: impl Fn(f32) -> f32) -> Result<Tensor> {
    if t.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op,
            why: "F32 only",
        });
    }
    if !t.is_contiguous() {
        return Err(Error::NotImplemented {
            op,
            why: "a contiguous layout is required",
        });
    }
    let Storage::Cpu(s) = t.storage() else {
        return Err(Error::NotImplemented {
            op,
            why: "CPU-only",
        });
    };
    let xs = s.as_f32_slice();
    let mut out = Vec::with_capacity(xs.len());
    for &x in xs {
        out.push(f(x));
    }
    Tensor::from_f32(out, *t.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn silu_known_values() {
        // silu(0) = 0
        // silu(1) ≈ 0.7311
        // silu(-1) ≈ -0.2689
        let t = Tensor::from_f32(
            alloc::vec![0.0, 1.0, -1.0],
            Shape::new(&[3]).unwrap(),
        )
        .unwrap();
        let s = silu(&t).unwrap();
        if let Storage::Cpu(st) = s.storage() {
            let v = st.as_f32_slice();
            assert!((v[0] - 0.0).abs() < 1e-6);
            assert!((v[1] - 0.7310586).abs() < 1e-5);
            assert!((v[2] - (-0.26894143)).abs() < 1e-5);
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn gelu_known_values() {
        // gelu(0) = 0
        // gelu(1) ≈ 0.84134
        let t = Tensor::from_f32(alloc::vec![0.0, 1.0], Shape::new(&[2]).unwrap()).unwrap();
        let g = gelu(&t).unwrap();
        if let Storage::Cpu(st) = g.storage() {
            let v = st.as_f32_slice();
            assert!((v[0] - 0.0).abs() < 1e-6);
            assert!((v[1] - 0.8413447).abs() < 1e-4);
        } else {
            panic!("expected CPU");
        }
    }
}
