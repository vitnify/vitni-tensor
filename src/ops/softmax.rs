//! Softmax along the last axis. Numerically stable: subtract max
//! before exp.
//!
//! M2: last-axis only. Arbitrary-dim softmax is a stride walk —
//! deferred until a model needs it.

use crate::{
    error::{Error, Result},
    storage::Storage,
    tensor::Tensor,
    DType,
};

/// `softmax(x)` along the last dimension. Returns a contiguous tensor
/// of the same shape. Numerically stable.
pub(crate) fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    if t.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op: "softmax",
            why: "M2 supports F32 only",
        });
    }
    if !t.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "softmax",
            why: "M2 requires contiguous layout",
        });
    }
    let Storage::Cpu(s) = t.storage() else {
        return Err(Error::NotImplemented {
            op: "softmax",
            why: "M2 is CPU-only",
        });
    };
    let xs = s.as_f32_slice();
    let shape = t.shape();
    let dims = shape.dims();
    if dims.is_empty() {
        return Err(Error::InvalidShape("softmax requires rank ≥ 1"));
    }
    let last = *dims.last().unwrap();
    let rows = xs.len() / last;

    let mut out = alloc::vec![0.0f32; xs.len()];
    for r in 0..rows {
        let row = &xs[r * last..(r + 1) * last];

        // max for numerical stability
        let mut mx = f32::NEG_INFINITY;
        for &v in row {
            if v > mx {
                mx = v;
            }
        }

        // exp + sum
        let mut sum = 0.0f32;
        let out_row = &mut out[r * last..(r + 1) * last];
        for (i, &v) in row.iter().enumerate() {
            let e = libm::expf(v - mx);
            out_row[i] = e;
            sum += e;
        }

        // normalize
        let inv = 1.0 / sum;
        for v in out_row.iter_mut() {
            *v *= inv;
        }
    }
    Tensor::from_f32(out, *shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn softmax_uniform() {
        let t = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[4]).unwrap()).unwrap();
        let s = softmax_last_dim(&t).unwrap();
        if let Storage::Cpu(st) = s.storage() {
            for v in st.as_f32_slice() {
                assert!((v - 0.25).abs() < 1e-6);
            }
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let t = Tensor::from_f32(
            alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(&[5]).unwrap(),
        )
        .unwrap();
        let s = softmax_last_dim(&t).unwrap();
        if let Storage::Cpu(st) = s.storage() {
            let sum: f32 = st.as_f32_slice().iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn softmax_2d_per_row() {
        // Two independent rows; each should sum to 1.
        let t = Tensor::from_f32(
            alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(&[2, 3]).unwrap(),
        )
        .unwrap();
        let s = softmax_last_dim(&t).unwrap();
        if let Storage::Cpu(st) = s.storage() {
            let v = st.as_f32_slice();
            let row0: f32 = v[0..3].iter().sum();
            let row1: f32 = v[3..6].iter().sum();
            assert!((row0 - 1.0).abs() < 1e-6);
            assert!((row1 - 1.0).abs() < 1e-6);
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn softmax_large_values_numerically_stable() {
        // Without max-subtract, exp(1000) overflows to inf.
        let t = Tensor::from_f32(
            alloc::vec![1000.0, 1000.0, 1000.0],
            Shape::new(&[3]).unwrap(),
        )
        .unwrap();
        let s = softmax_last_dim(&t).unwrap();
        if let Storage::Cpu(st) = s.storage() {
            for v in st.as_f32_slice() {
                assert!((v - 1.0 / 3.0).abs() < 1e-6);
            }
        } else {
            panic!("expected CPU");
        }
    }
}
