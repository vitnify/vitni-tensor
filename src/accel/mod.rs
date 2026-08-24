//! Accelerator — dispatch point for matmul-family ops.
//!
//! Same architectural pattern as `cert::sink`: vitni-tensor defines
//! the trait, downstream code (the host application) provides the impl
//! that calls `SYS_GPU_*` syscalls. This keeps vitni-tensor standalone
//! (no host-runtime dependency) while letting model code transparently route the
//! expensive ops through hardware acceleration.
//!
//! ## The determinism contract a GPU impl MUST honor
//!
//! A `SYS_GPU_MATMUL` implementation is only a valid verifier substitute if it
//! produces the SAME bits as the CPU canonical reduction (`ops::quant::
//! canonical_dot`). That is not automatic — stock cuBLAS / PyTorch-CUDA /
//! llama.cpp-CUDA choose their own reduction order and fuse multiply-add, and
//! do NOT match. The conforming reference kernels are in `kernels/`
//! (`canonical_matmul.metal`, `canonical_matmul.cu`): fixed reduction order
//! (element -> lane by `i % 8`, fixed pairwise tree), and NO fma contraction
//! (`--fmad=false` / `fastMathEnabled=false`, or the `__fmul_rn`/`__fadd_rn`
//! intrinsics). Verified bit-for-bit on Apple M3 Max (Metal) and NVIDIA T4
//! (CUDA) — see `kernels/README.md`; `tests/gpu_kernel_contract.rs` guards the
//! kernel algorithm against this crate in CI.
//!
//! ## Why on the model code, not on `Tensor`?
//!
//! `Tensor` storage stays immutable (`Arc<Storage>`) and lifetime-
//! free. If we put the accelerator on the tensor itself, every
//! Tensor would carry a backend reference and ops would need to
//! reconcile two operands' backends — Candle does this and it adds
//! significant lifetime + dispatch machinery.
//!
//! Instead, the FORWARD PASS owns the accelerator. `forward::step`
//! and `gemma::step` take `&mut impl Accelerator` and route each
//! matmul/softmax/rms_norm/silu through it. The accelerator decides
//! whether to actually GPU-dispatch or fall back to CPU based on op
//! shape (small matmuls stay CPU; large ones go GPU). This matches
//! the GPU/CPU crossover reality — see the project's
//! `gpu_crossover_bench` memory: GPU wins above ~5K dim, loses below.
//!
//! ## CPU vs GPU dispatch policy
//!
//! Even with a GPU accelerator registered, the model code routes
//! EVERY matmul through it. The accelerator itself decides what to
//! actually offload — typical policy:
//!
//!   - matmul with `min(m,n,k) < 512`: stay on CPU (overhead > gain)
//!   - matmul with any dim >= 512: SYS_GPU_MATMUL
//!   - softmax / rms_norm / silu: depends on tensor size
//!
//! `CpuAccelerator` always picks CPU; production `RuntimeGpu` would
//! threshold based on dim.
//!
//! ## runtime-side wiring sketch
//!
//! ```ignore
//! struct RuntimeGpu;
//! impl vitni_tensor::accel::Accelerator for RuntimeGpu {
//!     type Error = host::Error;
//!     fn matmul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Self::Error> {
//!         let (m, k_l) = (lhs.shape().dims()[0], lhs.shape().dims()[1]);
//!         let (k_r, n) = (rhs.shape().dims()[0], rhs.shape().dims()[1]);
//!         if m.min(n).min(k_l) < 512 {
//!             // Below crossover — CPU is faster.
//!             return Tensor::matmul(lhs, rhs).map_err(...);
//!         }
//!         // Upload, dispatch, download via the GPU backend::*
//!         let l_handle = host::syscall::gpu_alloc(m * k_l * 4)?;
//!         host::syscall::gpu_upload(l_handle, lhs_bytes(lhs))?;
//!         let r_handle = host::syscall::gpu_alloc(k_r * n * 4)?;
//!         host::syscall::gpu_upload(r_handle, rhs_bytes(rhs))?;
//!         let o_handle = host::syscall::gpu_alloc(m * n * 4)?;
//!         host::syscall::gpu_matmul(l_handle, r_handle, o_handle, m, n, k_l)?;
//!         let mut buf = vec![0u8; m * n * 4];
//!         host::syscall::gpu_download(o_handle, &mut buf)?;
//!         host::syscall::gpu_free(l_handle)?;
//!         host::syscall::gpu_free(r_handle)?;
//!         host::syscall::gpu_free(o_handle)?;
//!         Tensor::from_f32(f32_from_bytes(buf), Shape::new(&[m, n])?)
//!     }
//!     // similar for softmax_last_dim, rms_norm, silu
//! }
//! ```

use crate::{error::Error, tensor::Tensor};

/// Dispatch point for the matmul-family ops. The CPU fallback impl
/// (`CpuAccelerator`) always delegates to the tensor's existing
/// methods. A the GPU impl routes through `SYS_GPU_*` for large
/// shapes and falls back to CPU below the crossover.
pub trait Accelerator {
    /// Per-accelerator error type. Use `core::convert::Infallible`
    /// for CPU + recording sinks (both delegate to CPU, which only
    /// fails on shape errors that surface as `Error`).
    type Error: From<Error>;

    /// 2D matrix multiplication: `lhs @ rhs`. Shapes `[m, k] @ [k, n]`.
    fn matmul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Self::Error>;

    /// Softmax along the last dimension.
    fn softmax_last_dim(&mut self, t: &Tensor) -> Result<Tensor, Self::Error>;

    /// RMS-norm with per-feature scale.
    fn rms_norm(
        &mut self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, Self::Error>;

    /// SiLU activation.
    fn silu(&mut self, t: &Tensor) -> Result<Tensor, Self::Error>;
}

/// CPU-only accelerator. Delegates every op to the existing
/// `Tensor::*` methods. The default for callers who don't want GPU
/// routing.
#[derive(Debug, Default)]
pub struct CpuAccelerator;

impl Accelerator for CpuAccelerator {
    type Error = Error;

    fn matmul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Self::Error> {
        lhs.matmul(rhs)
    }
    fn softmax_last_dim(&mut self, t: &Tensor) -> Result<Tensor, Self::Error> {
        t.softmax_last_dim()
    }
    fn rms_norm(&mut self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, Self::Error> {
        x.rms_norm(weight, eps)
    }
    fn silu(&mut self, t: &Tensor) -> Result<Tensor, Self::Error> {
        t.silu()
    }
}

/// Recording accelerator: counts how many of each op were dispatched,
/// then delegates to CPU. Used by tests to verify the model code is
/// actually routing through the accelerator (and to validate per-
/// architecture op-count expectations — e.g., Llama2 with N layers
/// should dispatch exactly `4*N + 1` matmuls per decode step).
#[derive(Debug, Default)]
pub struct RecordingAccelerator {
    /// Number of `matmul` calls dispatched.
    pub matmul_count: usize,
    /// Number of `softmax_last_dim` calls dispatched.
    pub softmax_count: usize,
    /// Number of `rms_norm` calls dispatched.
    pub rms_norm_count: usize,
    /// Number of `silu` calls dispatched.
    pub silu_count: usize,
    /// Largest matmul shape seen, as `(m, n, k)`. Useful for
    /// confirming the GPU-threshold heuristic would trigger on
    /// realistic shapes.
    pub max_matmul: Option<(usize, usize, usize)>,
    inner: CpuAccelerator,
}

impl RecordingAccelerator {
    /// Total dispatched op count.
    pub fn total(&self) -> usize {
        self.matmul_count + self.softmax_count + self.rms_norm_count + self.silu_count
    }
}

impl Accelerator for RecordingAccelerator {
    type Error = Error;

    fn matmul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Self::Error> {
        self.matmul_count += 1;
        if lhs.shape().rank() == 2 && rhs.shape().rank() == 2 {
            let m = lhs.shape().dims()[0];
            let k = lhs.shape().dims()[1];
            let n = rhs.shape().dims()[1];
            self.max_matmul = Some(match self.max_matmul {
                None => (m, n, k),
                Some((om, on, ok)) => {
                    if (m * n * k) > (om * on * ok) {
                        (m, n, k)
                    } else {
                        (om, on, ok)
                    }
                }
            });
        }
        self.inner.matmul(lhs, rhs)
    }
    fn softmax_last_dim(&mut self, t: &Tensor) -> Result<Tensor, Self::Error> {
        self.softmax_count += 1;
        self.inner.softmax_last_dim(t)
    }
    fn rms_norm(&mut self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, Self::Error> {
        self.rms_norm_count += 1;
        self.inner.rms_norm(x, weight, eps)
    }
    fn silu(&mut self, t: &Tensor) -> Result<Tensor, Self::Error> {
        self.silu_count += 1;
        self.inner.silu(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn cpu_accelerator_matmul_matches_tensor_method() {
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
        let direct = a.matmul(&b).unwrap();
        let mut acc = CpuAccelerator;
        let via_accel = acc.matmul(&a, &b).unwrap();
        let unwrap = |t: &Tensor| {
            if let crate::storage::Storage::Cpu(s) = t.storage() {
                s.as_f32_slice().to_vec()
            } else {
                panic!()
            }
        };
        assert_eq!(unwrap(&direct), unwrap(&via_accel));
    }

    #[test]
    fn recording_accelerator_counts_ops() {
        let mut r = RecordingAccelerator::default();
        let t = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[4]).unwrap()).unwrap();
        let w = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[4]).unwrap()).unwrap();
        r.silu(&t).unwrap();
        r.silu(&t).unwrap();
        r.softmax_last_dim(&t).unwrap();
        r.rms_norm(&t, &w, 1e-5).unwrap();
        assert_eq!(r.silu_count, 2);
        assert_eq!(r.softmax_count, 1);
        assert_eq!(r.rms_norm_count, 1);
        assert_eq!(r.matmul_count, 0);
        assert_eq!(r.total(), 4);
    }

    #[test]
    fn recording_accelerator_tracks_max_matmul() {
        let mut r = RecordingAccelerator::default();
        let small_a = Tensor::from_f32(alloc::vec![1.0; 4], Shape::new(&[2, 2]).unwrap()).unwrap();
        let small_b = small_a.clone();
        r.matmul(&small_a, &small_b).unwrap();
        assert_eq!(r.max_matmul, Some((2, 2, 2)));

        let big_a = Tensor::from_f32(alloc::vec![1.0; 12], Shape::new(&[3, 4]).unwrap()).unwrap();
        let big_b = Tensor::from_f32(alloc::vec![1.0; 8], Shape::new(&[4, 2]).unwrap()).unwrap();
        r.matmul(&big_a, &big_b).unwrap();
        assert_eq!(r.max_matmul, Some((3, 2, 4)));
        assert_eq!(r.matmul_count, 2);
    }
}
