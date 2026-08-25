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

/// CPU storage: dtype-tagged, **4-byte-aligned** byte buffer.
///
/// Backed by `Vec<u32>` (alignment 4), not `Vec<u8>` (alignment 1), so the `&[f32]` and
/// `&[u32]` views are WELL-ALIGNED. Casting a `Vec<u8>`'s pointer to `*const f32` is
/// undefined behaviour whenever the allocation is not 4-aligned — miri flags it, and on
/// lenient hardware it merely happens to work while remaining UB the compiler may exploit.
/// `nbytes` is the logical byte length; the final word may be partially used for a
/// block-quantized dtype whose size is not a multiple of 4. Byte order is little-endian —
/// the crate's supported targets (x86-64, aarch64) are all LE, matching the on-wire GGUF
/// layout — the same assumption the previous `to_le_bytes` path already made.
#[derive(Debug)]
pub struct CpuStorage {
    words: Vec<u32>,
    nbytes: usize,
    dtype: DType,
}

impl CpuStorage {
    fn from_raw_bytes(bytes: &[u8], dtype: DType) -> Self {
        let nbytes = bytes.len();
        let mut words = alloc::vec![0u32; nbytes.div_ceil(4)];
        // Pack the bytes into the 4-aligned word buffer. SAFETY: `words` is 4-aligned, so
        // its pointer is valid for `words.len()*4 >= nbytes` byte writes.
        if nbytes != 0 {
            let dst = unsafe {
                core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 4)
            };
            dst[..nbytes].copy_from_slice(bytes);
        }
        Self { words, nbytes, dtype }
    }

    /// Allocate zero-initialized storage for `numel` elements of `dtype`.
    /// Works for both per-element and block-quantized dtypes via
    /// `DType::bytes_for_numel`.
    pub fn zeros(numel: usize, dtype: DType) -> Self {
        let nbytes = dtype.bytes_for_numel(numel);
        Self { words: alloc::vec![0u32; nbytes.div_ceil(4)], nbytes, dtype }
    }

    /// Construct from a raw byte blob already laid out in the dtype's
    /// on-wire format. Used by quantized weight loaders (GGUF, etc.)
    /// where the file stores the exact bytes we need to keep — no
    /// per-element reinterpretation is appropriate.
    pub fn from_bytes(bytes: Vec<u8>, dtype: DType) -> Self {
        Self::from_raw_bytes(&bytes, dtype)
    }

    /// Construct from an owned `Vec<f32>`. The dtype is fixed to `F32`.
    /// Stored as the f32 bit patterns in a 4-aligned word buffer, so `as_f32_slice`
    /// and `as_bytes` are both well-aligned and byte-identical to the LE layout.
    pub fn from_f32(data: Vec<f32>) -> Self {
        let words: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        let nbytes = words.len() * 4;
        Self { words, nbytes, dtype: DType::F32 }
    }

    /// Construct from an owned `Vec<u32>`. The dtype is fixed to `U32`.
    pub fn from_u32(data: Vec<u32>) -> Self {
        let nbytes = data.len() * 4;
        Self { words: data, nbytes, dtype: DType::U32 }
    }

    /// Read the dtype tag.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrow the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `words` is 4-aligned, so its pointer is valid for `nbytes` (<= len*4)
        // u8 reads (u8 needs align 1). A zero-length Vec still yields an aligned, non-null
        // pointer, valid for a 0-length slice.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.nbytes) }
    }

    /// Mutable borrow of the raw bytes. Used by ops that fill output
    /// tensors in place. Callers are responsible for honouring the
    /// dtype layout (e.g. an f32 op must write 4-byte aligned LE
    /// words; a Q4_0 op writes 18-byte blocks).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; the backing is 4-aligned so any f32 written here and read back
        // via `as_f32_slice` is well-aligned.
        unsafe { core::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.nbytes) }
    }

    /// View as `&[f32]` (requires `dtype == F32`).
    /// Caller must check dtype before calling.
    pub fn as_f32_slice(&self) -> &[f32] {
        debug_assert!(self.dtype == DType::F32, "as_f32_slice on non-F32 storage");
        debug_assert!(self.nbytes % 4 == 0, "F32 storage byte length not a multiple of 4");
        // SAFETY: `words` is a `Vec<u32>`, so its pointer has alignment 4 == align_of::<f32>()
        // and is valid for `nbytes/4` f32 reads. This is the alignment guarantee a `Vec<u8>`
        // backing did NOT provide.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr() as *const f32, self.nbytes / 4) }
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

    #[test]
    fn f32_view_is_4byte_aligned() {
        // The bug this fixes: a Vec<u8> backing gave a 1-aligned pointer, so the &[f32]
        // view was UB. The Vec<u32> backing must be 4-aligned.
        let s = CpuStorage::from_f32(alloc::vec![1.0; 7]);
        assert_eq!(s.as_f32_slice().as_ptr() as usize % 4, 0);
        // bytes round-trip through the LE layout unchanged
        assert_eq!(&s.as_bytes()[0..4], &1.0f32.to_le_bytes());
    }

    #[test]
    fn odd_length_quant_bytes_roundtrip() {
        // A block-quantized dtype's byte length need not be a multiple of 4.
        let raw: Vec<u8> = (0..18u8).collect();  // e.g. one Q4_0 block
        let s = CpuStorage::from_bytes(raw.clone(), DType::Q4_0);
        assert_eq!(s.as_bytes(), &raw[..]);
    }
}
