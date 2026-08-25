//! Llama2 model — the first end-to-end demonstration that ops in
//! `src/ops/*` compose into a real, runnable LLM architecture.
//!
//! Architecture follows karpathy's llama2.c (also what `the reference`
//! runs) so we have a known-good reference for cross-verification.
//!
//! ## Layout
//!
//! - `config` — model dimensions parsed from a llama2.c-format header
//! - `weights` — typed views over the binary weight blob, no copies
//! - `forward` — single-token forward pass returning logits, plus
//!   greedy-decode driver
//!
//! ## Status
//!
//! - Architecture defined; compiles clean no_std
//! - Verified bit-identical against an in-test reference impl on
//!   synthetic weights matching stories15M's shape
//! - Real stories15M weight load is the next plumbing step (host-side
//!   test reads the asset blob, runtime-side reads from P3 partition)

pub mod config;
pub mod forward;
pub mod forward_quantized;
pub mod gemma;
pub mod gguf;
pub mod inference;
pub mod quant_weights;
pub mod weights;

use crate::error::{Error, Result};

/// Reinterpret a byte view (a subslice of a weights blob / GGUF file) as `&[f32]`.
///
/// Weight loaders carve zero-copy `&[u8]` views out of a file blob at tensor offsets,
/// then need them as `&[f32]`. Casting `bytes.as_ptr() as *const f32` and building a
/// `&[f32]` is UB unless the pointer is 4-byte aligned — and neither the F32 dtype tag
/// nor the "format says tensors are aligned" contract *guarantees* that in Rust: a
/// `&[u8]` promises only 1-byte alignment, and a crafted or corrupt file can place an
/// F32 tensor at any offset. Well-formed GGUF pads tensor data to `general.alignment`
/// (>= 4) and real allocations are >= 16-aligned, so real models always pass; a
/// misaligned (malformed) input is rejected here instead of invoking UB.
pub(crate) fn f32_view(bytes: &[u8]) -> Result<&[f32]> {
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidShape("f32 view: byte length not a multiple of 4"));
    }
    if (bytes.as_ptr() as usize) % core::mem::align_of::<f32>() != 0 {
        return Err(Error::InvalidShape(
            "f32 view: tensor offset not 4-byte aligned (malformed file)",
        ));
    }
    // SAFETY: length is a multiple of 4 and the pointer is 4-byte aligned (both checked
    // above), so the whole region is a valid `[f32; len/4]`.
    Ok(unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) })
}

#[cfg(test)]
mod tests {
    use super::f32_view;
    use alloc::vec::Vec;

    #[test]
    fn f32_view_reads_aligned_bytes() {
        // A Vec<u32> is 4-aligned; view its bytes and reinterpret.
        let words: Vec<u32> = alloc::vec![1.0f32.to_bits(), (-2.5f32).to_bits()];
        let bytes =
            unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        let f = f32_view(bytes).expect("aligned bytes must succeed");
        assert_eq!(f, &[1.0f32, -2.5f32]);
    }

    #[test]
    fn f32_view_rejects_misaligned_offset() {
        // Take a 1-byte-offset subslice: guaranteed not 4-aligned. Must be rejected,
        // not turned into a misaligned &[f32] (the UB this guard prevents).
        let words: Vec<u32> = alloc::vec![0u32; 4];
        let base =
            unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        let misaligned = &base[1..9]; // length 8 (mult of 4) but offset +1
        assert!(f32_view(misaligned).is_err(), "misaligned offset must be rejected");
    }

    #[test]
    fn f32_view_rejects_non_multiple_of_4_len() {
        let words: Vec<u32> = alloc::vec![0u32; 2];
        let base =
            unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        assert!(f32_view(&base[..5]).is_err(), "length 5 must be rejected");
    }
}
