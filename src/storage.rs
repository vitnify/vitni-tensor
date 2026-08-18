//! Tensor storage — opaque per-device backing.
//!
//! `Cpu` storage is a dtype-tagged byte buffer in process address
//! space. `Gpu` storage is a kernel-managed handle (will be
//! `an opaque GPU backend handle` once we wire it). Storage is opaque
//! to ops; ops dispatch on `Device` and accept references.

use crate::dtype::DType;
use alloc::vec::Vec;

/// Backing storage for a tensor. One variant per device class.
#[derive(Debug)]
pub enum Storage {
    /// CPU-resident bytes. Tagged with dtype so reads know the
    /// element layout. We store bytes (not `Vec<T>`) so the same
    /// storage type holds any DType without generics propagating
    /// through the API.
    Cpu(CpuStorage),
    /// GPU-resident handle. Stub on host; on the target runtime this carries the
    /// `an opaque GPU backend handle`.
    Gpu(GpuStorage),
}

/// CPU storage: dtype-tagged byte buffer.
#[derive(Debug)]
pub struct CpuStorage {
    bytes: Vec<u8>,
    dtype: DType,
}

impl CpuStorage {
    /// Allocate zero-initialized storage for `numel` elements of `dtype`.
    /// Works for both per-element and block-quantized dtypes via
    /// `DType::bytes_for_numel`.
    pub fn zeros(numel: usize, dtype: DType) -> Self {
        let nbytes = dtype.bytes_for_numel(numel);
        Self {
            bytes: alloc::vec![0u8; nbytes],
            dtype,
        }
    }

    /// Construct from a raw byte blob already laid out in the dtype's
    /// on-wire format. Used by quantized weight loaders (GGUF, etc.)
    /// where the file stores the exact bytes we need to keep — no
    /// per-element reinterpretation is appropriate.
    pub fn from_bytes(bytes: Vec<u8>, dtype: DType) -> Self {
        Self { bytes, dtype }
    }

    /// Construct from an owned `Vec<f32>`. The dtype is fixed to `F32`.
    pub fn from_f32(data: Vec<f32>) -> Self {
        // Reinterpret without copy: f32 → bytes. We can do this
        // safely by transmuting via `Vec::into_raw_parts`-style
        // logic, but for clarity (and to keep `unsafe` localized)
        // we go via copy. Performance can be optimized later.
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            bytes,
            dtype: DType::F32,
        }
    }

    /// Construct from an owned `Vec<u32>`. The dtype is fixed to `U32`.
    pub fn from_u32(data: Vec<u32>) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            bytes,
            dtype: DType::U32,
        }
    }

    /// Read the dtype tag.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrow the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Mutable borrow of the raw bytes. Used by ops that fill output
    /// tensors in place. Callers are responsible for honouring the
    /// dtype layout (e.g. an f32 op must write 4-byte aligned LE
    /// words; a Q4_0 op writes 18-byte blocks).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// View as `&[f32]` (requires `dtype == F32`).
    /// Caller must check dtype before calling.
    pub fn as_f32_slice(&self) -> &[f32] {
        debug_assert!(self.dtype == DType::F32, "as_f32_slice on non-F32 storage");
        debug_assert!(self.bytes.len() % 4 == 0, "F32 storage not 4-byte aligned");
        // SAFETY: dtype-tagged, aligned by construction (Vec<u8> from
        // `to_le_bytes` writes).
        unsafe {
            core::slice::from_raw_parts(
                self.bytes.as_ptr() as *const f32,
                self.bytes.len() / 4,
            )
        }
    }
}

/// GPU storage: kernel handle. Stubbed for host builds; on the target runtime this
/// carries the `an opaque GPU backend handle` (handle + shape).
#[derive(Debug)]
pub struct GpuStorage {
    /// Kernel-assigned tensor handle.
    pub handle: u32,
    /// Element count (so we can validate ops without consulting kernel).
    pub numel: usize,
    /// dtype tag.
    pub dtype: DType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_zeros_sized_correctly() {
        let s = CpuStorage::zeros(10, DType::F32);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.as_bytes().len(), 40);
        assert!(s.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn cpu_from_f32_roundtrips() {
        let s = CpuStorage::from_f32(alloc::vec![1.0, -2.5, 3.14]);
        assert_eq!(s.dtype(), DType::F32);
        let view = s.as_f32_slice();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0], 1.0);
        assert_eq!(view[1], -2.5);
        assert!((view[2] - 3.14).abs() < 1e-6);
    }
}
