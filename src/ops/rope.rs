//! Rotary Position Embedding (RoPE).
//!
//! Used by Llama/Mistral/Phi/Qwen/Gemma for query and key tensors
//! before attention. Encodes position via complex-plane rotations:
//! pairs of consecutive features `(x_{2i}, x_{2i+1})` are treated as
//! `(real, imag)` and rotated by angle `m * theta^(-2i/dim)` where
//! `m` is the absolute token position and `dim` is the head dim.
//!
//! At phase 2 we apply RoPE in-place-style to one tensor of shape
//! `[seq, n_heads, head_dim]` for a single sequence. Pre-computed
//! sin/cos cache support is the GPU path.

use crate::{
    error::{Error, Result},
    storage::Storage,
    tensor::Tensor,
    DType,
};
use alloc::vec::Vec;

/// Apply RoPE to a `[seq, n_heads, head_dim]` F32 tensor.
///
/// `theta`: base frequency (Llama-2 uses 10000.0, Llama-3 uses
///   500000.0; pass whatever the model spec demands).
/// `position_offset`: absolute position of the first token in the
///   input (for KV-cache continuations). Pass 0 for fresh prompts.
///
/// `head_dim` must be even (RoPE rotates pairs).
pub(crate) fn rope(t: &Tensor, theta: f32, position_offset: usize) -> Result<Tensor> {
    if t.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op: "rope",
            why: "F32 only",
        });
    }
    if !t.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "rope",
            why: "a contiguous layout is required",
        });
    }
    if t.shape().rank() != 3 {
        return Err(Error::ShapeMismatch {
            expected: "[seq, n_heads, head_dim]",
            got: "non-rank-3 input",
        });
    }
    let dims = t.shape().dims();
    let (seq, n_heads, head_dim) = (dims[0], dims[1], dims[2]);
    if head_dim % 2 != 0 {
        return Err(Error::InvalidShape("rope: head_dim must be even"));
    }
    let Storage::Cpu(s) = t.storage() else {
        return Err(Error::NotImplemented {
            op: "rope",
            why: "CPU-only",
        });
    };
    let xs = s.as_f32_slice();
    let mut out = alloc::vec![0.0f32; xs.len()];

    let half = head_dim / 2;
    // Pre-compute inverse frequencies: 1 / theta^(2i/dim) for i in [0, half).
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2 * i) as f32 / head_dim as f32;
        inv_freq.push(1.0 / libm::powf(theta, exponent));
    }

    for s_idx in 0..seq {
        let pos = (position_offset + s_idx) as f32;
        // Cosines and sines for this position. Compute once per token,
        // reuse across heads.
        let mut cos_cache = Vec::with_capacity(half);
        let mut sin_cache = Vec::with_capacity(half);
        for i in 0..half {
            let angle = pos * inv_freq[i];
            cos_cache.push(libm::cosf(angle));
            sin_cache.push(libm::sinf(angle));
        }

        for h_idx in 0..n_heads {
            let head_off = (s_idx * n_heads + h_idx) * head_dim;
            for i in 0..half {
                let a = xs[head_off + 2 * i];
                let b = xs[head_off + 2 * i + 1];
                let c = cos_cache[i];
                let s_ = sin_cache[i];
                // (a, b) -> (a*c - b*s, a*s + b*c)
                out[head_off + 2 * i] = a * c - b * s_;
                out[head_off + 2 * i + 1] = a * s_ + b * c;
            }
        }
    }

    Tensor::from_f32(out, *t.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn rope_position_zero_is_identity() {
        // At position 0, all angles are 0 → cos=1, sin=0 → no rotation.
        let shape = Shape::new(&[1, 1, 4]).unwrap();
        let t = Tensor::from_f32(alloc::vec![1.0, 2.0, 3.0, 4.0], shape).unwrap();
        let r = rope(&t, 10000.0, 0).unwrap();
        if let Storage::Cpu(s) = r.storage() {
            let v = s.as_f32_slice();
            assert!((v[0] - 1.0).abs() < 1e-6);
            assert!((v[1] - 2.0).abs() < 1e-6);
            assert!((v[2] - 3.0).abs() < 1e-6);
            assert!((v[3] - 4.0).abs() < 1e-6);
        } else {
            panic!("expected CPU");
        }
    }

    #[test]
    fn rope_odd_head_dim_errors() {
        let shape = Shape::new(&[1, 1, 3]).unwrap();
        let t = Tensor::from_f32(alloc::vec![1.0, 2.0, 3.0], shape).unwrap();
        assert!(rope(&t, 10000.0, 0).is_err());
    }

    #[test]
    fn rope_shape_preserved() {
        let shape = Shape::new(&[2, 3, 4]).unwrap();
        let t = Tensor::from_f32(alloc::vec![1.0; 24], shape).unwrap();
        let r = rope(&t, 10000.0, 0).unwrap();
        assert_eq!(r.shape().dims(), shape.dims());
    }

    #[test]
    fn rope_preserves_l2_norm() {
        // Rotation is unitary: ||rope(x)||² == ||x||².
        let shape = Shape::new(&[3, 2, 4]).unwrap();
        let xs: alloc::vec::Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let l2: f32 = xs.iter().map(|x| x * x).sum();
        let t = Tensor::from_f32(xs, shape).unwrap();
        let r = rope(&t, 10000.0, 5).unwrap();
        if let Storage::Cpu(s) = r.storage() {
            let l2_after: f32 = s.as_f32_slice().iter().map(|x| x * x).sum();
            assert!((l2 - l2_after).abs() < 1e-3, "L2 norm not preserved: {l2} vs {l2_after}");
        } else {
            panic!("expected CPU");
        }
    }
}
