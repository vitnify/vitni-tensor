//! Element types for `Tensor`. Subset of Candle's set — we add as
//! models demand.
//!
//! Block-quantized dtypes (Q4_0, Q8_0) match the GGML/llama.cpp
//! formats so weight blobs can be read directly from GGUF without
//! repacking. Per-element sizing doesn't apply; use
//! `bytes_for_numel(n)` instead of `size_in_bytes()`.

/// Supported element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float (half precision).
    F16,
    /// bfloat16 (1 sign + 8 exp + 7 mantissa).
    BF16,
    /// Unsigned 32-bit integer (used for token IDs, indices).
    U32,
    /// Signed 64-bit integer (used for shape arithmetic, position).
    I64,
    /// GGML Q4_0 — 32-element blocks of 4-bit weights, one f16
    /// scale per block. Layout per block (18 bytes for 32
    /// elements): `[scale: u16 LE f16][qs: u8; 16]`. Each `qs`
    /// byte packs two 4-bit signed weights as `(low | high << 4)`.
    /// Effective storage: 4.5 bits per weight. Matches the
    /// `SYS_GPU_Q4_LINEAR` kernel format exactly.
    Q4_0,
    /// GGML Q8_0 — 32-element blocks of int8 weights, one f16
    /// scale per block. Layout per block (34 bytes for 32
    /// elements): `[scale: u16 LE f16][qs: i8; 32]`. Effective
    /// storage: 8.5 bits per weight.
    Q8_0,
    /// GGML Q4_K — K-quant 256-element super-blocks with 8
    /// sub-blocks of 32 weights each, super-block d + dmin (f16),
    /// per-sub-block 6-bit scale + min (packed in 12 bytes), then
    /// 128 bytes of packed 4-bit quants. Layout per super-block
    /// (144 bytes for 256 elements):
    ///   `[d: u16 f16][dmin: u16 f16][scales: u8; 12][qs: u8; 128]`
    /// Effective storage: 4.5 bits per weight. Standard format
    /// for the bulk of weights in Mistral 7B / Llama 2 Q4_K_M.
    Q4_K,
    /// GGML Q6_K — K-quant 256-element super-blocks with 16
    /// sub-blocks of 16 weights each, super-block d (f16), per-
    /// sub-block i8 scale, then 6-bit quants split across ql/qh
    /// arrays. Layout per super-block (210 bytes for 256 elements):
    ///   `[ql: u8; 128][qh: u8; 64][scales: i8; 16][d: u16 f16]`
    /// Effective storage: 6.5625 bits per weight. Standard format
    /// for output projection (lm_head) in Mistral 7B Q4_K_M.
    Q6_K,
}

/// Q4_K super-block: 256 weights per block.
pub const Q4_K_BLOCK_NUMEL: usize = 256;
/// Q4_K bytes per super-block: 4 (f16 d + dmin) + 12 (scales/mins) + 128 (qs).
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Q6_K super-block: 256 weights per block.
pub const Q6_K_BLOCK_NUMEL: usize = 256;
/// Q6_K bytes per super-block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (f16 d).
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Q4_0 block size in elements.
pub const Q4_0_BLOCK_NUMEL: usize = 32;
/// Q4_0 bytes per block: 2 (f16 scale) + 16 (packed nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Q8_0 block size in elements.
pub const Q8_0_BLOCK_NUMEL: usize = 32;
/// Q8_0 bytes per block: 2 (f16 scale) + 32 (int8 weights).
pub const Q8_0_BLOCK_BYTES: usize = 34;

impl DType {
    /// Size of one element in bytes for fixed-size dtypes. Panics
    /// for block-quantized dtypes — use `bytes_for_numel` instead.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 => 8,
            DType::Q4_0 | DType::Q8_0 | DType::Q4_K | DType::Q6_K => {
                // Per-element sizing doesn't apply to block-quantized
                // dtypes. Treat as a programmer error caught at panic
                // time so we hear about it instead of computing the
                // wrong byte count silently.
                panic!("size_in_bytes() on block-quantized DType — use bytes_for_numel()")
            }
        }
    }

    /// Total bytes needed to store `numel` elements of this dtype.
    /// Handles both per-element and block-quantized layouts.
    /// For block dtypes, `numel` MUST be a multiple of the block size.
    pub const fn bytes_for_numel(self, numel: usize) -> usize {
        match self {
            DType::F32 | DType::U32 => numel * 4,
            DType::F16 | DType::BF16 => numel * 2,
            DType::I64 => numel * 8,
            DType::Q4_0 => {
                // The kernel SYS_GPU_Q4_LINEAR layout assumes whole
                // blocks. Caller is responsible for padding model
                // weights to a multiple of 32.
                (numel / Q4_0_BLOCK_NUMEL) * Q4_0_BLOCK_BYTES
            }
            DType::Q8_0 => (numel / Q8_0_BLOCK_NUMEL) * Q8_0_BLOCK_BYTES,
            DType::Q4_K => (numel / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES,
            DType::Q6_K => (numel / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES,
        }
    }

    /// `true` if this dtype stores weights as block-quantized blobs.
    /// Quantized dtypes are weight-only — no quantized intermediate
    /// activations. Op dispatch checks this to decide whether to
    /// route through `linear_q4_0` (matmul wrapping kernel
    /// `SYS_GPU_Q4_LINEAR`) vs the generic f32 matmul path.
    pub const fn is_block_quantized(self) -> bool {
        matches!(self, DType::Q4_0 | DType::Q8_0 | DType::Q4_K | DType::Q6_K)
    }

    /// Block size in elements (1 for non-quantized dtypes).
    pub const fn block_numel(self) -> usize {
        match self {
            DType::Q4_0 => Q4_0_BLOCK_NUMEL,
            DType::Q8_0 => Q8_0_BLOCK_NUMEL,
            _ => 1,
        }
    }

    /// Human-readable tag for error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::Q4_0 => "q4_0",
            DType::Q8_0 => "q8_0",
            DType::Q4_K => "q4_k",
            DType::Q6_K => "q6_k",
        }
    }
}
