//! Embedding lookup. `indices[i] -> weight[indices[i]]`.
//!
//! `indices`: rank-N U32 tensor of token IDs.
//! `weight`: rank-2 F32 tensor of shape `[vocab_size, dim]`.
//! Output: rank-(N+1) F32 tensor — the index shape with `dim` appended.
//!
//! For the common LLM case is indices=`[seq]`, weight=`[vocab, dim]`,
//! output=`[seq, dim]`. We support exactly that shape; higher ranks
//! later.

use crate::{
    error::{Error, Result},
    storage::Storage,
    tensor::Tensor,
    DType, Shape,
};

/// Look up each `indices[i]` row of `weight`.
///
/// Returns a tensor of shape `[indices.len(), dim]` where `dim` is
/// the last dimension of `weight`.
pub(crate) fn embedding(indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if indices.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            expected: "u32",
            got: indices.dtype().as_str(),
        });
    }
    if weight.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch {
            expected: "f32",
            got: weight.dtype().as_str(),
        });
    }
    if weight.shape().rank() != 2 {
        return Err(Error::InvalidShape("embedding: weight must be rank-2"));
    }
    if indices.shape().rank() != 1 {
        return Err(Error::NotImplemented {
            op: "embedding",
            why: "only rank-1 indices are supported",
        });
    }
    if !indices.is_contiguous() || !weight.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "embedding",
            why: "a contiguous layout is required",
        });
    }

    let (vocab, dim) = (weight.shape().dims()[0], weight.shape().dims()[1]);
    let seq = indices.shape().dims()[0];

    let (Storage::Cpu(ix_s), Storage::Cpu(w_s)) = (indices.storage(), weight.storage()) else {
        return Err(Error::NotImplemented {
            op: "embedding",
            why: "CPU-only",
        });
    };

    // U32 indices reinterpreted from byte storage. We don't expose a
    // generic `as_u32_slice` on CpuStorage yet, so reconstitute here.
    let ix_bytes = ix_s.as_bytes();
    let w = w_s.as_f32_slice();

    let mut out = alloc::vec![0.0f32; seq * dim];
    for s in 0..seq {
        let ix_bytes_slice: &[u8; 4] = ix_bytes[s * 4..s * 4 + 4]
            .try_into()
            .expect("U32 storage 4-byte aligned");
        let ix = u32::from_le_bytes(*ix_bytes_slice) as usize;
        if ix >= vocab {
            return Err(Error::InvalidShape("embedding: index out of vocab range"));
        }
        let src = &w[ix * dim..(ix + 1) * dim];
        let dst = &mut out[s * dim..(s + 1) * dim];
        dst.copy_from_slice(src);
    }

    Tensor::from_f32(out, Shape::new(&[seq, dim])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_picks_rows() {
        // vocab=3, dim=2
        // weight = [[10,11], [20,21], [30,31]]
        let w = Tensor::from_f32(
            alloc::vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            Shape::new(&[3, 2]).unwrap(),
        )
        .unwrap();
        // indices = [2, 0, 1] -> [[30,31],[10,11],[20,21]]
        let ix = Tensor::from_u32(alloc::vec![2, 0, 1], Shape::new(&[3]).unwrap()).unwrap();
        let out = embedding(&ix, &w).unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);
        if let Storage::Cpu(s) = out.storage() {
            assert_eq!(
                s.as_f32_slice(),
                &[30.0, 31.0, 10.0, 11.0, 20.0, 21.0]
            );
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn embedding_out_of_range_errors() {
        let w = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[2, 2]).unwrap()).unwrap();
        let ix = Tensor::from_u32(alloc::vec![5], Shape::new(&[1]).unwrap()).unwrap();
        assert!(embedding(&ix, &w).is_err());
    }
}
