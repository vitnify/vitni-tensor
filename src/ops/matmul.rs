//! 2D matrix multiplication.
//!
//! CPU reduction uses the CANONICAL fixed shape (CANON_BLOCK-sized blocks
//! then a fixed pairwise tree) — the same regime as the reference implementation and the
//! CUDA reference kernel, so issuer and verifier agree bit-for-bit. Hardware
//! may parallelize the shape but may not choose its own order.
//! M3: route to `SYS_GPU_MATMUL` when the tensors live on a Gpu device.
//!
//! Higher-rank batched matmul is M3+ work. For M2 we accept rank-2
//! tensors only — that's enough to walk a single transformer
//! layer's projections (Q/K/V/O/MLP) one at a time.

use crate::{
    error::{Error, Result},
    shape::Shape,
    storage::Storage,
    tensor::Tensor,
    DType,
};

/// Canonical reduction block size. This is part of the numerical CONTRACT,
/// not a tuning knob: it must match every other implementation in the system
/// (the reference implementation's row dot, the CUDA reference kernel) or results are
/// deterministic per-shape but not comparable ACROSS shapes — which would
/// break replay between the issuer and the verifier.
pub(crate) const CANON_BLOCK: usize = 8;

/// `lhs @ rhs` for rank-2 tensors. Result shape `[m, n]` where
/// `lhs` is `[m, k]` and `rhs` is `[k, n]`.
pub(crate) fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if !lhs.device().same(rhs.device()) {
        return Err(Error::DeviceMismatch { op: "matmul" });
    }
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        return Err(Error::NotImplemented {
            op: "matmul",
            why: "M2 supports F32 only",
        });
    }
    if lhs.shape().rank() != 2 || rhs.shape().rank() != 2 {
        return Err(Error::NotImplemented {
            op: "matmul",
            why: "M2 supports rank-2 inputs only (batched matmul = M3)",
        });
    }
    if !lhs.is_contiguous() || !rhs.is_contiguous() {
        return Err(Error::NotImplemented {
            op: "matmul",
            why: "M2 requires contiguous layout",
        });
    }

    let l_dims = lhs.shape().dims();
    let r_dims = rhs.shape().dims();
    let (m, k_l) = (l_dims[0], l_dims[1]);
    let (k_r, n) = (r_dims[0], r_dims[1]);
    if k_l != k_r {
        return Err(Error::ShapeMismatch {
            expected: "lhs cols == rhs rows",
            got: "k mismatch",
        });
    }
    let k = k_l;

    let Storage::Cpu(ls) = lhs.storage() else {
        return Err(Error::NotImplemented {
            op: "matmul",
            why: "M2 is CPU-only",
        });
    };
    let Storage::Cpu(rs) = rhs.storage() else {
        return Err(Error::NotImplemented {
            op: "matmul",
            why: "M2 is CPU-only",
        });
    };
    let a = ls.as_f32_slice();
    let b = rs.as_f32_slice();

    // CANONICAL REDUCTION over k — the same fixed shape used by
    // the reference implementation's row dot and by the CUDA reference kernel, so every
    // path in the system shares ONE numerical regime.
    //
    // This replaced a serial `for kk in 0..k { acc += .. }` whose determinism
    // was a property of the loop shape — a convention any optimizer could
    // silently break. The canonical form makes the shape explicit AND is
    // faster: a serial accumulator is a loop-carried dependency that blocks
    // vectorization, whereas independent blocks vectorize freely (measured
    // 2.9-8x on the standalone benchmark).
    //
    // Verified bit-identical on Apple M3 Max (aarch64), AMD x86_64 and an
    // NVIDIA Tesla T4 (hash 40ce39e1 on the reference vector); the control,
    // a hardware-chosen atomic order, differed by one ULP and was not even
    // reproducible run-to-run on the same GPU.
    // ONE reduction implementation for the whole crate.
    //
    // This used to carry its own inline copy of the canonical shape. Two
    // copies of a numerical contract is two things to keep in step, and they
    // did fall out of step: when the quantized path moved to lane-pinned v2
    // this file was still computing v1, so `vitnium-receipt-verifier`'s regime
    // probe (which runs through matmul) certified a regime the GGUF replay
    // path no longer used. Delegating removes the failure mode entirely.
    //
    // `b`'s column is strided by `n`, so it is gathered into a contiguous
    // scratch first — a copy, but this is the f32 fallback path, not the
    // quantized hot path, and agreeing with the contract beats being quick.
    let mut out = alloc::vec![0.0f32; m * n];
    let mut col = alloc::vec![0.0f32; k];
    for j in 0..n {
        if k == 0 {
            for i in 0..m {
                out[i * n + j] = 0.0;
            }
            continue;
        }
        if n == 1 {
            col.copy_from_slice(&b[..k]);
        } else {
            for kk in 0..k {
                col[kk] = b[kk * n + j];
            }
        }
        for i in 0..m {
            let row = &a[i * k..i * k + k];
            out[i * n + j] = crate::ops::quant::canonical_dot(row, &col, k);
        }
    }
    Tensor::from_f32(out, Shape::new(&[m, n])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_MATMUL_HASH: u64 = 0x8a42_8433_686d_13af;

    /// PINS THE EXACT BITS of the matmul reduction.
    ///
    /// This crate is what excert-verifier replays through, so its arithmetic
    /// IS the certificate's definition of "the same computation". The reduction
    /// is the CANONICAL fixed shape (CANON_BLOCK blocks + pairwise tree), the
    /// same one used by the reference implementation and the CUDA reference kernel, so all
    /// three share one numerical regime.
    /// That is not hypothetical: the reference implementation carried an x86-only SSE
    /// `sse_row_dot` (4 lane accumulators + horizontal tree) alongside a serial
    /// fallback, so the SAME model produced DIFFERENT BITS per architecture and
    /// its certificates could never have replayed cross-arch. Nobody noticed,
    /// because nothing checked.
    ///
    /// This test is the check. If someone vectorizes, threads, reassociates, or
    /// enables FMA contraction in this reduction, the hash moves and this fails
    /// LOUDLY — instead of quietly invalidating every previously issued ExCert.
    ///
    /// If you are here because this test failed and the change was deliberate:
    /// changing the reduction changes the numerical regime, so previously
    /// issued certificates will no longer replay. That needs a recorded regime
    /// version in the certificate, not a silent hash update here.
    #[test]
    fn matmul_reduction_bits_are_pinned() {
        // Deterministic inputs — same LCG as the cross-vendor contract test.
        let (m, k, n) = (4usize, 64usize, 4usize);
        let mut s: u64 = 0x1234;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
        };
        let a: alloc::vec::Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: alloc::vec::Vec<f32> = (0..k * n).map(|_| rnd()).collect();

        let ta = Tensor::from_f32(a, Shape::new(&[m, k]).unwrap()).unwrap();
        let tb = Tensor::from_f32(b, Shape::new(&[k, n]).unwrap()).unwrap();
        let out = matmul(&ta, &tb).unwrap();
        let Storage::Cpu(os) = out.storage() else { panic!("expected cpu storage") };
        let v = os.as_f32_slice();

        // FNV-1a over the raw f32 bit patterns: any change in rounding or
        // accumulation order moves this.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for x in v.iter() {
            for byte in x.to_bits().to_le_bytes().iter() {
                h ^= *byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }

        // Baseline for the CANONICAL regime (2026-07-29). Superseded the
        // serial-ijk baseline 0x5f47f0ef98b0ab4c; certificates issued under
        // that older regime will not replay against this build.
        assert_eq!(
            h, PINNED_MATMUL_HASH,
            "matmul reduction bits changed (got {:#018x}, expected {:#018x}) — \
             this silently invalidates every previously issued ExCert; see the \
             doc comment on this test",
            h, PINNED_MATMUL_HASH
        );
    }

    #[test]
    fn identity_matmul() {
        // [2,2] @ I[2,2] = [2,2] same
        let a = Tensor::from_f32(
            alloc::vec![1.0, 2.0, 3.0, 4.0],
            Shape::new(&[2, 2]).unwrap(),
        )
        .unwrap();
        let i = Tensor::from_f32(
            alloc::vec![1.0, 0.0, 0.0, 1.0],
            Shape::new(&[2, 2]).unwrap(),
        )
        .unwrap();
        let c = matmul(&a, &i).unwrap();
        if let Storage::Cpu(s) = c.storage() {
            assert_eq!(s.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("expected CPU storage");
        }
    }

    #[test]
    fn known_2x3_x_3x2() {
        // a = [[1,2,3],[4,5,6]]  (2x3)
        // b = [[7,8],[9,10],[11,12]]  (3x2)
        // c = a @ b = [[58, 64], [139, 154]]
        let a = Tensor::from_f32(
            alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(&[2, 3]).unwrap(),
        )
        .unwrap();
        let b = Tensor::from_f32(
            alloc::vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::new(&[3, 2]).unwrap(),
        )
        .unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        if let Storage::Cpu(s) = c.storage() {
            assert_eq!(s.as_f32_slice(), &[58.0, 64.0, 139.0, 154.0]);
        } else {
            panic!("expected CPU storage");
        }
    }

    #[test]
    fn k_mismatch_errors() {
        let a = Tensor::from_f32(alloc::vec![1.0; 6], Shape::new(&[2, 3]).unwrap()).unwrap();
        let b = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[2, 2]).unwrap()).unwrap();
        assert!(matmul(&a, &b).is_err());
    }
}
