//! Element-wise binary ops: add, sub, mul, div.
//!
//! No broadcasting at M2 — shapes must match exactly. Add broadcasting
//! when a model architecture demands it (most LLM ops are same-shape).

use crate::{
    error::{Error, Result},
    storage::Storage,
    tensor::Tensor,
    DType,
};
use alloc::vec::Vec;

fn check_compatible(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    if !lhs.device().same(rhs.device()) {
        return Err(Error::DeviceMismatch { op });
    }
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::DTypeMismatch {
            expected: lhs.dtype().as_str(),
            got: rhs.dtype().as_str(),
        });
    }
    if lhs.shape().dims() != rhs.shape().dims() {
        return Err(Error::ShapeMismatch {
            expected: "matching shape",
            got: "different shape",
        });
    }
    Ok(())
}

/// Elementwise `lhs[i] + rhs[i]`. Returns a new contiguous tensor.
pub(crate) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    check_compatible("add", lhs, rhs)?;
    apply_f32(lhs, rhs, |a, b| a + b)
}

/// Elementwise `lhs[i] - rhs[i]`.
pub(crate) fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    check_compatible("sub", lhs, rhs)?;
    apply_f32(lhs, rhs, |a, b| a - b)
}

/// Elementwise `lhs[i] * rhs[i]`.
pub(crate) fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    check_compatible("mul", lhs, rhs)?;
    apply_f32(lhs, rhs, |a, b| a * b)
}

/// Elementwise `lhs[i] / rhs[i]`. No zero-check (matches IEEE 754
/// behavior — `1.0 / 0.0 == inf`).
pub(crate) fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    check_compatible("div", lhs, rhs)?;
    apply_f32(lhs, rhs, |a, b| a / b)
}

/// Common F32 elementwise pathway. CPU contiguous fast path; other
/// layouts handled by reading via strides (M5 work).
fn apply_f32(lhs: &Tensor, rhs: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    if lhs.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op: "binary",
            why: "M2 supports F32 only",
        });
    }
    if !lhs.is_contiguous() || !rhs.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "binary",
            why: "M2 requires contiguous layout",
        });
    }
    let (Storage::Cpu(l), Storage::Cpu(r)) = (lhs.storage(), rhs.storage()) else {
        return Err(Error::NotImplemented {
            op: "binary",
            why: "M2 is CPU-only",
        });
    };
    let l = l.as_f32_slice();
    let r = r.as_f32_slice();
    let mut out = Vec::with_capacity(l.len());
    for (&a, &b) in l.iter().zip(r.iter()) {
        out.push(f(a, b));
    }
    Tensor::from_f32(out, *lhs.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn add_works() {
        let a = Tensor::from_f32(alloc::vec![1.0, 2.0, 3.0], Shape::new(&[3]).unwrap()).unwrap();
        let b = Tensor::from_f32(alloc::vec![10.0, 20.0, 30.0], Shape::new(&[3]).unwrap()).unwrap();
        let c = add(&a, &b).unwrap();
        if let Storage::Cpu(s) = c.storage() {
            assert_eq!(s.as_f32_slice(), &[11.0, 22.0, 33.0]);
        } else {
            panic!("expected CPU storage");
        }
    }

    #[test]
    fn mul_works() {
        let a = Tensor::from_f32(alloc::vec![2.0, 3.0], Shape::new(&[2]).unwrap()).unwrap();
        let b = Tensor::from_f32(alloc::vec![4.0, 5.0], Shape::new(&[2]).unwrap()).unwrap();
        let c = mul(&a, &b).unwrap();
        if let Storage::Cpu(s) = c.storage() {
            assert_eq!(s.as_f32_slice(), &[8.0, 15.0]);
        } else {
            panic!("expected CPU storage");
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = Tensor::from_f32(alloc::vec![1.0, 2.0], Shape::new(&[2]).unwrap()).unwrap();
        let b = Tensor::from_f32(alloc::vec![1.0, 2.0, 3.0], Shape::new(&[3]).unwrap()).unwrap();
        assert!(add(&a, &b).is_err());
    }
}
