//! The `Tensor` type — storage + shape + strides + dtype + device.
//!
//! Constructors and accessors at M1. Ops (matmul, softmax, rms_norm,
//! embedding, rope, ...) live in `src/ops/*` and are exposed as
//! methods at the bottom of this file.

use crate::{
    device::Device,
    dtype::DType,
    error::{Error, Result},
    ops,
    shape::{Shape, Strides},
    storage::{CpuStorage, Storage},
};
use alloc::{sync::Arc, vec::Vec};

/// Multi-dimensional tensor. `Arc`-wrapped storage so reshape/view
/// ops are zero-copy.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Shape,
    strides: Strides,
    offset: usize,
    dtype: DType,
    device: Device,
}

impl Tensor {
    /// Create a zero-filled CPU tensor of the given shape and dtype.
    pub fn zeros(shape: Shape, dtype: DType) -> Result<Self> {
        let storage = Storage::Cpu(CpuStorage::zeros(shape.numel(), dtype));
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage: Arc::new(storage),
            shape,
            strides,
            offset: 0,
            dtype,
            device: Device::Cpu,
        })
    }

    /// Create a CPU F32 tensor from a `Vec<f32>` and an explicit shape.
    /// Errors if `data.len()` doesn't match `shape.numel()`.
    pub fn from_f32(data: Vec<f32>, shape: Shape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(Error::InvalidShape(
                "from_f32: data.len() != shape.numel()",
            ));
        }
        let storage = Storage::Cpu(CpuStorage::from_f32(data));
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage: Arc::new(storage),
            shape,
            strides,
            offset: 0,
            dtype: DType::F32,
            device: Device::Cpu,
        })
    }

    /// Create a CPU U32 tensor from a `Vec<u32>` and an explicit shape.
    /// Used for token IDs and indices.
    pub fn from_u32(data: Vec<u32>, shape: Shape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(Error::InvalidShape(
                "from_u32: data.len() != shape.numel()",
            ));
        }
        let storage = Storage::Cpu(CpuStorage::from_u32(data));
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage: Arc::new(storage),
            shape,
            strides,
            offset: 0,
            dtype: DType::U32,
            device: Device::Cpu,
        })
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Strides (in elements, not bytes) for indexing.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Element dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Where the storage lives.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Total element count.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Borrow the storage (for ops that need direct access).
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Borrow the underlying byte buffer if storage is CPU. Errors if
    /// the tensor lives on GPU — callers needing GPU bytes should
    /// route through the `Accelerator` API instead.
    pub fn storage_cpu_bytes(&self) -> Result<&[u8]> {
        match &*self.storage {
            Storage::Cpu(c) => Ok(c.as_bytes()),
            Storage::Gpu(_) => Err(Error::Internal(
                "storage_cpu_bytes called on GPU tensor",
            )),
        }
    }

    /// `true` if storage is laid out contiguously (row-major,
    /// matching `shape.contiguous_strides()`).
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape) && self.offset == 0
    }
}

// ============================================================================
// Op methods. Each delegates to `ops/*` so the impl stays focused.
// API mirrors Candle's so model definitions port mechanically.
// ============================================================================

impl Tensor {
    /// Element-wise addition. Same shape required.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        ops::binary::add(self, other)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        ops::binary::sub(self, other)
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        ops::binary::mul(self, other)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        ops::binary::div(self, other)
    }

    /// 2D matrix multiplication. `self @ other`.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        ops::matmul::matmul(self, other)
    }

    /// SiLU activation: `x * sigmoid(x)`.
    pub fn silu(&self) -> Result<Tensor> {
        ops::unary::silu(self)
    }

    /// GELU activation (exact, erf-based).
    pub fn gelu(&self) -> Result<Tensor> {
        ops::unary::gelu(self)
    }

    /// Softmax along the last dimension.
    pub fn softmax_last_dim(&self) -> Result<Tensor> {
        ops::softmax::softmax_last_dim(self)
    }

    /// RMS normalization with learnable per-feature scale.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops::rms_norm::rms_norm(self, weight, eps)
    }

    /// Embedding lookup. `self` is rank-1 U32 indices, `weight` is
    /// rank-2 F32 `[vocab, dim]`. Returns `[seq, dim]`.
    pub fn embedding(&self, weight: &Tensor) -> Result<Tensor> {
        ops::embedding::embedding(self, weight)
    }

    /// Apply Rotary Position Embedding. `self` is `[seq, n_heads, head_dim]`.
    pub fn rope(&self, theta: f32, position_offset: usize) -> Result<Tensor> {
        ops::rope::rope(self, theta, position_offset)
    }

    /// Argmax along the last dimension. Returns U32 indices.
    /// Used for greedy decoding.
    pub fn argmax_last_dim(&self) -> Result<Tensor> {
        ops::argmax::argmax_last_dim(self)
    }

    /// Linear layer with Q4_0-quantized weights. `self` is the F32
    /// input `[batch, in_feat]`; `w` is a Q4_0 tensor of shape
    /// `[out_feat, in_feat]` laid out in GGML block format. Returns
    /// F32 `[batch, out_feat]`.
    ///
    /// Routes through the kernel `SYS_GPU_Q4_LINEAR` when an
    /// `Accelerator` is selected; falls back to `linear_q4_0_cpu`
    /// otherwise. Either path produces bit-identical f32 output
    /// for the same Q4_0 input (deterministic dequant + matmul).
    pub fn linear_q4_0(&self, w: &Tensor, out_feat: usize) -> Result<Tensor> {
        use crate::shape::Shape;
        use alloc::vec;

        if self.dtype() != DType::F32 {
            return Err(Error::Internal("linear_q4_0: input must be F32"));
        }
        if w.dtype() != DType::Q4_0 {
            return Err(Error::Internal("linear_q4_0: weight must be Q4_0"));
        }
        let x_shape = self.shape().dims();
        if x_shape.len() != 2 {
            return Err(Error::Internal("linear_q4_0: input must be 2D"));
        }
        let batch = x_shape[0];
        let in_feat = x_shape[1];

        // Pull byte views; ops::quant operates on raw slices to keep
        // the GPU path symmetric (kernel syscall takes raw pointers).
        let x_bytes = self.storage_cpu_bytes()?;
        let w_bytes = w.storage_cpu_bytes()?;

        // SAFETY: x_bytes is F32 by dtype check above.
        let x_f32: &[f32] = unsafe {
            core::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, x_bytes.len() / 4)
        };

        // Allocate output as Vec<f32>; the op fills it in place. We
        // construct the Tensor from this vec at the end — avoids the
        // Arc<Storage> mutability problem (Tensor::zeros wraps storage
        // in Arc, no &mut access from there).
        let mut y_f32 = vec![0f32; batch * out_feat];
        ops::quant::linear_q4_0_cpu(x_f32, w_bytes, &mut y_f32, batch, in_feat, out_feat)?;

        Tensor::from_f32(y_f32, Shape::new(&[batch, out_feat])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_constructs() {
        let s = Shape::new(&[2, 3]).unwrap();
        let t = Tensor::zeros(s, DType::F32).unwrap();
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.device(), Device::Cpu);
        assert!(t.is_contiguous());
    }

    #[test]
    fn from_f32_round_trips() {
        let s = Shape::new(&[2, 2]).unwrap();
        let t = Tensor::from_f32(alloc::vec![1.0, 2.0, 3.0, 4.0], s).unwrap();
        assert_eq!(t.numel(), 4);
        if let Storage::Cpu(cpu) = t.storage() {
            let view = cpu.as_f32_slice();
            assert_eq!(view, &[1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("expected CPU storage");
        }
    }

    #[test]
    fn shape_mismatch_errors() {
        let s = Shape::new(&[2, 2]).unwrap();
        let res = Tensor::from_f32(alloc::vec![1.0, 2.0, 3.0], s); // 3 != 4
        assert!(res.is_err());
    }
}
