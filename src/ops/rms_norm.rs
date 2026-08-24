//! RMS normalization. Llama/Mistral/Phi/Qwen all use this.
//!
//! `rms_norm(x, w, eps) = (x / sqrt(mean(x^2) + eps)) * w`
//!
//! Normalization is along the last dimension; `w` is a per-feature
//! scale vector of length equal to the last dim.

use crate::{
    error::{Error, Result},
    storage::Storage,
    tensor::Tensor,
    DType,
};

/// Apply RMS-norm to `x` with learnable scale `weight` and epsilon.
///
/// `x`: any rank ≥ 1, F32, contiguous.
/// `weight`: rank-1, F32, length == x.shape().dims().last().
/// `eps`: small constant for numerical stability (typically 1e-5).
pub(crate) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op: "rms_norm",
            why: "F32 only",
        });
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "rms_norm",
            why: "a contiguous layout is required",
        });
    }
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(Error::InvalidShape("rms_norm requires rank ≥ 1"));
    }
    let feat = *dims.last().unwrap();
    if weight.shape().rank() != 1 || weight.shape().dims()[0] != feat {
        return Err(Error::ShapeMismatch {
            expected: "weight length == last dim of x",
            got: "weight mismatch",
        });
    }
    let Storage::Cpu(ws) = weight.storage() else {
        return Err(Error::NotImplemented {
            op: "rms_norm",
            why: "CPU-only",
        });
    };
    rms_norm_slice(x, ws.as_f32_slice(), eps)
}

/// Slice-weight core of [`rms_norm`].
///
/// The weight is read only as `&[f32]`, so requiring a `Tensor` forced callers
/// holding a borrowed weight to `.to_vec()` it — a 16 KB copy plus an
/// allocation per norm, per layer, per token (96 of them per token on
/// Mistral-7B, for data that never changes). Bit-identical: same arithmetic in
/// the same order, only the plumbing differs.
pub(crate) fn rms_norm_slice(x: &Tensor, ws: &[f32], eps: f32) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op: "rms_norm",
            why: "F32 only",
        });
    }
    if !x.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "rms_norm",
            why: "a contiguous layout is required",
        });
    }
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(Error::InvalidShape("rms_norm requires rank ≥ 1"));
    }
    let feat = *dims.last().unwrap();
    if ws.len() != feat {
        return Err(Error::ShapeMismatch {
            expected: "weight length == last dim of x",
            got: "weight mismatch",
        });
    }
    let Storage::Cpu(xs) = x.storage() else {
        return Err(Error::NotImplemented {
            op: "rms_norm",
            why: "CPU-only",
        });
    };
    let xs = xs.as_f32_slice();
    let rows = xs.len() / feat;

    let mut out = alloc::vec![0.0f32; xs.len()];
    for r in 0..rows {
        let row = &xs[r * feat..(r + 1) * feat];
        // mean(x^2) via the crate's ONE canonical reduction (lane-pinned +
        // fixed tree) — the same shape the matmul uses. sum(x*x) == dot(x, x),
        // so this is a `canonical_dot` of the row with itself: bit-identical
        // across vector width / thread count / GPU, and parallelizable (unlike
        // the serial accumulator it replaced, which pinned it to one thread).
        let sumsq = crate::ops::quant::canonical_dot(row, row, feat);
        let mean = sumsq / feat as f32;
        let scale = 1.0 / libm::sqrtf(mean + eps);

        let out_row = &mut out[r * feat..(r + 1) * feat];
        for (i, &v) in row.iter().enumerate() {
            out_row[i] = v * scale * ws[i];
        }
    }
    Tensor::from_f32(out, *x.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn rms_norm_identity_scale() {
        // weight = [1,1,1,1], so output = x / rms(x).
        // x = [1,1,1,1] → rms = 1, output = [1,1,1,1].
        let x = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[4]).unwrap()).unwrap();
        let w = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[4]).unwrap()).unwrap();
        let y = rms_norm(&x, &w, 0.0).unwrap();
        if let Storage::Cpu(s) = y.storage() {
            for v in s.as_f32_slice() {
                assert!((v - 1.0).abs() < 1e-6);
            }
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn rms_norm_known() {
        // x = [3, 4], rms = sqrt(12.5) ≈ 3.5355
        // output = [3, 4] / 3.5355 * [1, 1] ≈ [0.8485, 1.1314]
        let x = Tensor::from_f32(alloc::vec![3.0, 4.0], Shape::new(&[2]).unwrap()).unwrap();
        let w = Tensor::from_f32(alloc::vec![1.0, 1.0], Shape::new(&[2]).unwrap()).unwrap();
        let y = rms_norm(&x, &w, 0.0).unwrap();
        if let Storage::Cpu(s) = y.storage() {
            let v = s.as_f32_slice();
            assert!((v[0] - 0.8485281).abs() < 1e-4);
            assert!((v[1] - 1.1313708).abs() < 1e-4);
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn rms_norm_weighted() {
        // weight scales each feature.
        let x = Tensor::from_f32(alloc::vec![1.0, 1.0], Shape::new(&[2]).unwrap()).unwrap();
        let w = Tensor::from_f32(alloc::vec![2.0, 3.0], Shape::new(&[2]).unwrap()).unwrap();
        let y = rms_norm(&x, &w, 0.0).unwrap();
        if let Storage::Cpu(s) = y.storage() {
            let v = s.as_f32_slice();
            // rms = 1, so output = [1*2, 1*3] = [2, 3]
            assert!((v[0] - 2.0).abs() < 1e-6);
            assert!((v[1] - 3.0).abs() < 1e-6);
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn rms_norm_weight_mismatch_errors() {
        let x = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[4]).unwrap()).unwrap();
        let w = Tensor::from_f32(alloc::vec![1.0; 3], Shape::new(&[3]).unwrap()).unwrap();
        assert!(rms_norm(&x, &w, 1e-5).is_err());
    }
}
