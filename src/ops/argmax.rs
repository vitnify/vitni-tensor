//! Argmax along the last dimension. Returns U32 indices.
//!
//! Used for greedy decoding: `logits.argmax_last_dim()` picks the
//! highest-probability token at each position.
//!
//! Determinism note: when two values tie, we keep the FIRST one
//! (smallest index). This matches the reference's behavior and
//! ensures cross-vendor reproducibility — without this rule a
//! reordered traversal could pick a different tied value.

use crate::{
    error::{Error, Result},
    shape::Shape,
    storage::Storage,
    tensor::Tensor,
    DType,
};

/// Argmax along the last dimension. Input must be F32. Output is
/// U32 with shape `input.shape()[:-1]`.
pub(crate) fn argmax_last_dim(t: &Tensor) -> Result<Tensor> {
    if t.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch {
            expected: "f32",
            got: t.dtype().as_str(),
        });
    }
    if !t.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "argmax",
            why: "a contiguous layout is required",
        });
    }
    let Storage::Cpu(s) = t.storage() else {
        return Err(Error::NotImplemented {
            op: "argmax",
            why: "CPU-only",
        });
    };
    let xs = s.as_f32_slice();
    let dims = t.shape().dims();
    if dims.is_empty() {
        return Err(Error::InvalidShape("argmax requires rank ≥ 1"));
    }
    let last = *dims.last().unwrap();
    let rows = xs.len() / last;

    let mut out = alloc::vec::Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &xs[r * last..(r + 1) * last];
        let mut best_idx = 0usize;
        let mut best_val = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        out.push(best_idx as u32);
    }

    if rows == 1 {
        Tensor::from_u32(out, Shape::new(&[1])?)
    } else {
        let out_dims: alloc::vec::Vec<usize> = dims[..dims.len() - 1].to_vec();
        Tensor::from_u32(out, Shape::new(&out_dims)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_simple() {
        let t = Tensor::from_f32(
            alloc::vec![0.1, 0.5, 0.3, 0.8, 0.2],
            Shape::new(&[5]).unwrap(),
        )
        .unwrap();
        let a = argmax_last_dim(&t).unwrap();
        if let Storage::Cpu(s) = a.storage() {
            let v: alloc::vec::Vec<u32> = (0..1)
                .map(|i| {
                    let b: [u8; 4] = s.as_bytes()[i * 4..(i + 1) * 4].try_into().unwrap();
                    u32::from_le_bytes(b)
                })
                .collect();
            assert_eq!(v, alloc::vec![3]);
        }
    }

    #[test]
    fn argmax_tie_picks_first() {
        // Two 0.5s — must pick index 1 (the first occurrence).
        let t = Tensor::from_f32(
            alloc::vec![0.1, 0.5, 0.3, 0.5, 0.2],
            Shape::new(&[5]).unwrap(),
        )
        .unwrap();
        let a = argmax_last_dim(&t).unwrap();
        if let Storage::Cpu(s) = a.storage() {
            let b: [u8; 4] = s.as_bytes()[0..4].try_into().unwrap();
            assert_eq!(u32::from_le_bytes(b), 1);
        }
    }

    #[test]
    fn argmax_per_row() {
        // [2, 3] input → argmax over last dim → [2] output
        let t = Tensor::from_f32(
            alloc::vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.4],
            Shape::new(&[2, 3]).unwrap(),
        )
        .unwrap();
        let a = argmax_last_dim(&t).unwrap();
        assert_eq!(a.shape().dims(), &[2]);
        if let Storage::Cpu(s) = a.storage() {
            let v0 = u32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap());
            let v1 = u32::from_le_bytes(s.as_bytes()[4..8].try_into().unwrap());
            assert_eq!((v0, v1), (1, 0));
        }
    }
}
