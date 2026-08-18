//! GGUF v3 reader — parses the file format used by llama.cpp / Ollama
//! to distribute quantized LLM weights.
//!
//! Spec: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! ## File layout
//!
//! ```text
//!   magic              "GGUF"           4 bytes
//!   version            uint32 LE        3
//!   tensor_count       uint64 LE
//!   metadata_kv_count  uint64 LE
//!   metadata[ ]        kv pairs
//!   tensor_infos[ ]    name + shape + dtype + offset
//!   padding            zeros to alignment (default 32, set via
//!                      general.alignment metadata u32)
//!   tensor_data        opaque blob; each tensor at base + offset
//! ```
//!
//! ## Supported tensor dtypes (M2 / Phase 2)
//!
//! - F32 (ggml type 0)
//! - F16 (ggml type 1)  — read as raw u16 LE, callers dequant if needed
//! - Q4_0 (ggml type 2) — passes straight through to vitni-tensor's
//!                        `DType::Q4_0` (same byte layout)
//! - Q8_0 (ggml type 8) — same, `DType::Q8_0`
//!
//! Unsupported: Q4_1, Q5_*, Q8_1, Q*_K (k-quants). These are common
//! in distributed checkpoints (Mistral 7B Q4_K_M uses Q4_K + Q6_K).
//! Adding them is mechanical — define the block layout, add the
//! dequant op — but out of scope for the initial parser.
//!
//! ## No_std discipline
//!
//! We never copy tensor bytes — every `GgufTensor` carries a slice
//! into the input blob. The owner holds the blob (via `include_bytes!`
//! on host tests, `cas_file_read` on the target runtime) and the GgufFile borrows
//! from it.

use alloc::borrow::ToOwned;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};

// ---------------------------------------------------------------
// GGUF constants — match `ggml.h` / `gguf.md` exactly.
// ---------------------------------------------------------------

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];
const DEFAULT_ALIGNMENT: u64 = 32;
/// Smallest a tensor-info record can be on disk: an empty name
/// (`u64` length + 0 bytes) + `n_dims` (`u32`, may be 0) + type
/// (`u32`) + offset (`u64`) = 24 bytes. Used to bound the pre-read
/// capacity reservation against an attacker-inflated tensor count.
const MIN_TENSOR_INFO_BYTES: u64 = 24;

/// GGUF metadata value type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgufValueType {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::UINT8,
            1 => Self::INT8,
            2 => Self::UINT16,
            3 => Self::INT16,
            4 => Self::UINT32,
            5 => Self::INT32,
            6 => Self::FLOAT32,
            7 => Self::BOOL,
            8 => Self::STRING,
            9 => Self::ARRAY,
            10 => Self::UINT64,
            11 => Self::INT64,
            12 => Self::FLOAT64,
            _ => return None,
        })
    }
}

/// GGML tensor element type tags. Only those we plan to consume are
/// listed in the body of `GgufTensorType`; everything else parses
/// into `Other(raw)` so we can read the file's index without dying
/// on an unsupported tensor type — the caller decides what to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    /// 5-bit block quant, 32-element blocks, 22 bytes/block. The bulk
    /// of tensors in Qwen2.5 GGUFs (q4_k_m keeps most weights as Q5_0).
    Q5_0,
    /// K-quant 256-element super-block, 4.5 bits/weight. Used for
    /// the bulk of weights in Mistral 7B / Llama 2 Q4_K_M.
    Q4_K,
    /// K-quant 256-element super-block, 6.5625 bits/weight. Used
    /// for output projection (lm_head) in Mistral 7B Q4_K_M.
    Q6_K,
    /// Anything we don't natively support (Q4_1, Q5_*, Q2_K, ...).
    /// Carries the raw u32 from the file for diagnostics + future
    /// expansion. Callers should error out cleanly on these.
    Other(u32),
}

impl GgufTensorType {
    fn from_u32(v: u32) -> Self {
        // ggml_type values from llama.cpp's ggml.h.
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            6 => Self::Q5_0,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            14 => Self::Q6_K,
            other => Self::Other(other),
        }
    }

    /// Compute the byte size of a tensor with `numel` elements at
    /// this dtype. For block-quantized types this rounds up to a
    /// whole block — caller is responsible for that being the
    /// actual on-disk size (GGUF guarantees it).
    fn bytes_for_numel(self, numel: u64) -> Result<u64> {
        Ok(match self {
            Self::F32 => numel.checked_mul(4).ok_or(Error::Overflow(
                "gguf: F32 byte size overflow",
            ))?,
            Self::F16 => numel.checked_mul(2).ok_or(Error::Overflow(
                "gguf: F16 byte size overflow",
            ))?,
            Self::Q4_0 => {
                if numel % 32 != 0 {
                    return Err(Error::Malformed(
                        "gguf: Q4_0 tensor numel not a multiple of 32",
                    ));
                }
                (numel / 32).checked_mul(18).ok_or(Error::Overflow(
                    "gguf: Q4_0 byte size overflow",
                ))?
            }
            Self::Q8_0 => {
                if numel % 32 != 0 {
                    return Err(Error::Malformed(
                        "gguf: Q8_0 tensor numel not a multiple of 32",
                    ));
                }
                (numel / 32).checked_mul(34).ok_or(Error::Overflow(
                    "gguf: Q8_0 byte size overflow",
                ))?
            }
            Self::Q5_0 => {
                if numel % 32 != 0 {
                    return Err(Error::Malformed(
                        "gguf: Q5_0 tensor numel not a multiple of 32",
                    ));
                }
                (numel / 32).checked_mul(22).ok_or(Error::Overflow(
                    "gguf: Q5_0 byte size overflow",
                ))?
            }
            Self::Q4_K => {
                if numel % 256 != 0 {
                    return Err(Error::Malformed(
                        "gguf: Q4_K tensor numel not a multiple of 256",
                    ));
                }
                (numel / 256).checked_mul(144).ok_or(Error::Overflow(
                    "gguf: Q4_K byte size overflow",
                ))?
            }
            Self::Q6_K => {
                if numel % 256 != 0 {
                    return Err(Error::Malformed(
                        "gguf: Q6_K tensor numel not a multiple of 256",
                    ));
                }
                (numel / 256).checked_mul(210).ok_or(Error::Overflow(
                    "gguf: Q6_K byte size overflow",
                ))?
            }
            Self::Other(raw) => {
                let _ = raw;
                return Err(Error::Malformed(
                    "gguf: byte size unknown for unsupported ggml_type",
                ));
            }
        })
    }
}

// ---------------------------------------------------------------
// Parsed structures
// ---------------------------------------------------------------

/// A single key-value metadata entry. Strings and primitives only —
/// arrays are stored as `GgufValue::Array(GgufArray)` which we also
/// only decode lazily (the storage carries the raw bytes + element
/// type so callers that need it can decode without a redundant
/// parse pass).
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(GgufArray),
}

/// A typed array of GGUF primitives. We don't unpack into a `Vec`
/// of typed values — for token-vocab metadata that's 32K+ strings
/// per file and we'd pay an allocation per entry. Instead store the
/// raw byte range; callers index into it on demand.
#[derive(Debug, Clone)]
pub struct GgufArray {
    /// Element type (one of the primitives — nested arrays not allowed).
    pub elem_type: GgufValueType,
    /// Number of elements.
    pub len: u64,
    /// Raw byte slice. For fixed-size primitives this is `len *
    /// sizeof(elem)`; for strings it's a packed sequence of
    /// `(u64 length, bytes)` records.
    pub data: Vec<u8>,
}

/// Tensor metadata: name, shape, dtype, and a slice into the
/// underlying file blob holding the tensor's bytes.
#[derive(Debug, Clone)]
pub struct GgufTensor<'a> {
    /// Tensor name as stored in the file. Conventional layouts use
    /// dotted paths like `"blk.0.attn_q.weight"`, `"token_embd.weight"`,
    /// `"output_norm.weight"`. The mapping from these names to the
    /// `Weights` struct's fields is the loader's responsibility (see
    /// `gguf_to_llama_weights` in a future commit).
    pub name: String,
    /// Shape in GGML order (note: GGML stores shapes reversed from
    /// what most NumPy-style frameworks expect — caller may need to
    /// reverse).
    pub shape: Vec<u64>,
    /// Element type.
    pub dtype: GgufTensorType,
    /// Raw byte slice into the input blob. Lives as long as `'a`.
    pub bytes: &'a [u8],
}

impl GgufTensor<'_> {
    /// Total element count = product of shape. Saturates rather than
    /// panicking on overflow: `parse` only admits tensors whose shape
    /// product fits u64 (see `checked_numel`), so for any tensor that
    /// came out of a parsed file this is the exact product; the
    /// saturating fold just keeps a hand-built shape from panicking a
    /// debug build.
    pub fn numel(&self) -> u64 {
        self.shape.iter().fold(1u64, |acc, &d| acc.saturating_mul(d))
    }
}

/// Parsed GGUF file. Tensors and metadata reference the input blob;
/// owner must keep it alive for `'a`.
#[derive(Debug)]
pub struct GgufFile<'a> {
    /// Format version (2 or 3).
    pub version: u32,
    /// All key-value metadata, indexed by key.
    pub metadata: BTreeMap<String, GgufValue>,
    /// Tensors, ordered as in the file.
    pub tensors: Vec<GgufTensor<'a>>,
    /// `name → index in tensors` for O(log n) lookup.
    pub by_name: BTreeMap<String, usize>,
}

impl<'a> GgufFile<'a> {
    /// Parse a GGUF file from an in-memory blob. Zero-copy — the
    /// returned `GgufFile` borrows from `blob` for the lifetime
    /// of `'a`.
    pub fn parse(blob: &'a [u8]) -> Result<Self> {
        let mut cur = Cursor::new(blob);

        // Header.
        let magic = cur.read_n(4)?;
        if magic != GGUF_MAGIC {
            return Err(Error::Malformed("gguf: bad magic (not 'GGUF')"));
        }
        let version = cur.read_u32()?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(Error::Malformed(
                "gguf: unsupported version (only v2/v3 supported)",
            ));
        }
        let tensor_count = cur.read_u64()?;
        let metadata_kv_count = cur.read_u64()?;

        // Metadata.
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let value_type_raw = cur.read_u32()?;
            let value_type = GgufValueType::from_u32(value_type_raw)
                .ok_or(Error::Malformed("gguf: unknown value type"))?;
            let value = cur.read_value(value_type)?;
            metadata.insert(key, value);
        }

        // Tensor infos.
        // Stored back-to-back: each is `name + n_dims + dims[] + type + offset`.
        // After reading all of them we pad to alignment, then the
        // tensor data section begins.
        // `tensor_count` is attacker-controlled; feeding it straight to
        // `with_capacity` lets a corrupt file (e.g. u64::MAX) abort the
        // process on a giant allocation before we read a single record.
        // Cap the reservation by the fewest bytes a tensor-info record
        // can occupy (empty name len + n_dims + type + offset = 24). The
        // loop still reads and validates every record and errors out on
        // EOF, so an inflated count just yields a `Truncated` error.
        let remaining = blob.len().saturating_sub(cur.pos) as u64;
        let cap = core::cmp::min(tensor_count, remaining / MIN_TENSOR_INFO_BYTES + 1) as usize;
        let mut tensor_infos: Vec<(String, Vec<u64>, GgufTensorType, u64)> =
            Vec::with_capacity(cap);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            if n_dims > 8 {
                return Err(Error::Malformed(
                    "gguf: tensor rank > 8 (sanity bound)",
                ));
            }
            let mut shape: Vec<u64> = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cur.read_u64()?);
            }
            let raw_dtype = cur.read_u32()?;
            let dtype = GgufTensorType::from_u32(raw_dtype);
            let offset = cur.read_u64()?;
            tensor_infos.push((name, shape, dtype, offset));
        }

        // Alignment.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                GgufValue::U32(n) => Some(*n as u64),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);

        // `alignment` comes straight from `general.alignment`; a zero
        // would divide-by-zero in `div_ceil` below, and the rounded-up
        // result is fed to a checked multiply so a huge value can't wrap.
        if alignment == 0 {
            return Err(Error::Malformed("gguf: general.alignment is zero"));
        }
        let unaligned = cur.pos as u64;
        let aligned = unaligned
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or(Error::Overflow("gguf: alignment padding overflow"))?;
        cur.pos = aligned as usize;
        let tensor_data_base = cur.pos;

        // Materialize tensor byte slices.
        let mut tensors: Vec<GgufTensor<'a>> = Vec::with_capacity(tensor_infos.len());
        let mut by_name = BTreeMap::new();
        for (i, (name, shape, dtype, offset)) in tensor_infos.into_iter().enumerate() {
            let numel = checked_numel(&shape)?;
            let nbytes = dtype.bytes_for_numel(numel)?;
            let start = tensor_data_base
                .checked_add(offset as usize)
                .ok_or(Error::Overflow("gguf: tensor offset overflow"))?;
            let end = start
                .checked_add(nbytes as usize)
                .ok_or(Error::Overflow("gguf: tensor end overflow"))?;
            if end > blob.len() {
                return Err(Error::Truncated(
                    "gguf: tensor extends past end of file",
                ));
            }
            let bytes = &blob[start..end];
            by_name.insert(name.clone(), i);
            tensors.push(GgufTensor {
                name,
                shape,
                dtype,
                bytes,
            });
        }

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            by_name,
        })
    }

    /// Look up a tensor by name. Returns `None` if absent.
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor<'a>> {
        // invariant: by_name only ever holds indices `parse` produced
        // from `tensors.len()`, so the index is always in bounds.
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Convenience: read a u32 metadata value or error.
    pub fn metadata_u32(&self, key: &str) -> Result<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(v)) => Ok(*v),
            Some(_) => Err(Error::Malformed("gguf: metadata is not U32")),
            None => Err(Error::Malformed("gguf: metadata key missing")),
        }
    }

    /// Convenience: read a u64 metadata value or error.
    pub fn metadata_u64(&self, key: &str) -> Result<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::U64(v)) => Ok(*v),
            Some(GgufValue::U32(v)) => Ok(*v as u64),
            Some(_) => Err(Error::Malformed("gguf: metadata is not U64")),
            None => Err(Error::Malformed("gguf: metadata key missing")),
        }
    }

    /// Convenience: read a string metadata value or error.
    pub fn metadata_str(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Ok(s.as_str()),
            Some(_) => Err(Error::Malformed("gguf: metadata is not String")),
            None => Err(Error::Malformed("gguf: metadata key missing")),
        }
    }

    /// Raw metadata value accessor (for reading arrays like the tokenizer vocab).
    pub fn metadata_value(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }
}

/// Product of a tensor's shape dims with overflow checking. Dims are
/// `u64` straight from the file, so a corrupt shape can name a product
/// that wraps `u64` (and panics a debug build's `Iterator::product`).
/// Fold with `checked_mul` so it surfaces as `Error::Overflow` instead.
fn checked_numel(shape: &[u64]) -> Result<u64> {
    let mut numel: u64 = 1;
    for &dim in shape {
        numel = numel
            .checked_mul(dim)
            .ok_or(Error::Overflow("gguf: shape product overflows u64"))?;
    }
    Ok(numel)
}

// ---------------------------------------------------------------
// Cursor — local helper, no I/O traits in no_std.
// ---------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_n(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Overflow(
            "gguf: read past end (overflow)",
        ))?;
        if end > self.buf.len() {
            return Err(Error::Truncated("gguf: read past end"));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32> {
        // invariant: read_n returns exactly 4 bytes or Err, so s[0..4] hold.
        let s = self.read_n(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        // invariant: read_n returns exactly 8 bytes or Err, so s[0..8] hold.
        let s = self.read_n(8)?;
        Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }

    fn read_string(&mut self) -> Result<String> {
        let n = self.read_u64()? as usize;
        if n > 1 << 20 {
            // 1 MiB sanity cap on string lengths — keys/values
            // realistically max around 64 KiB; this stops a
            // corrupted file from triggering a huge allocation.
            return Err(Error::Malformed("gguf: string length > 1 MiB"));
        }
        let bytes = self.read_n(n)?;
        let s = core::str::from_utf8(bytes)
            .map_err(|_| Error::Malformed("gguf: non-UTF8 string"))?;
        Ok(s.to_owned())
    }

    fn read_value(&mut self, ty: GgufValueType) -> Result<GgufValue> {
        Ok(match ty {
            GgufValueType::UINT8 => GgufValue::U8(self.read_n(1)?[0]),
            GgufValueType::INT8 => GgufValue::I8(self.read_n(1)?[0] as i8),
            GgufValueType::UINT16 => {
                let s = self.read_n(2)?;
                GgufValue::U16(u16::from_le_bytes([s[0], s[1]]))
            }
            GgufValueType::INT16 => {
                let s = self.read_n(2)?;
                GgufValue::I16(i16::from_le_bytes([s[0], s[1]]))
            }
            GgufValueType::UINT32 => GgufValue::U32(self.read_u32()?),
            GgufValueType::INT32 => GgufValue::I32(self.read_u32()? as i32),
            GgufValueType::FLOAT32 => GgufValue::F32(f32::from_bits(self.read_u32()?)),
            GgufValueType::BOOL => GgufValue::Bool(self.read_n(1)?[0] != 0),
            GgufValueType::STRING => GgufValue::String(self.read_string()?),
            GgufValueType::UINT64 => GgufValue::U64(self.read_u64()?),
            GgufValueType::INT64 => GgufValue::I64(self.read_u64()? as i64),
            GgufValueType::FLOAT64 => {
                let bits = self.read_u64()?;
                GgufValue::F64(f64::from_bits(bits))
            }
            GgufValueType::ARRAY => {
                let elem_type_raw = self.read_u32()?;
                let elem_type = GgufValueType::from_u32(elem_type_raw)
                    .ok_or(Error::Malformed("gguf: bad array elem type"))?;
                if matches!(elem_type, GgufValueType::ARRAY) {
                    return Err(Error::Malformed(
                        "gguf: nested arrays not supported",
                    ));
                }
                let len = self.read_u64()?;
                // Skip ahead by computing total byte length without
                // decoding individual elements. For fixed-size primitives
                // it's len * sizeof; for strings it's variable.
                let start = self.pos;
                if matches!(elem_type, GgufValueType::STRING) {
                    for _ in 0..len {
                        let n = self.read_u64()? as usize;
                        self.pos = self.pos.checked_add(n).ok_or(
                            Error::Overflow("gguf: array string overflow"),
                        )?;
                        if self.pos > self.buf.len() {
                            return Err(Error::Truncated(
                                "gguf: array string past end",
                            ));
                        }
                    }
                } else {
                    let elem_size = match elem_type {
                        GgufValueType::UINT8 | GgufValueType::INT8 | GgufValueType::BOOL => 1,
                        GgufValueType::UINT16 | GgufValueType::INT16 => 2,
                        GgufValueType::UINT32 | GgufValueType::INT32 | GgufValueType::FLOAT32 => 4,
                        GgufValueType::UINT64 | GgufValueType::INT64 | GgufValueType::FLOAT64 => 8,
                        // STRING is handled above; ARRAY is rejected above.
                        // Any other tag never reaches here, but return an
                        // error rather than panic to keep the parser
                        // panic-free on every input path.
                        _ => return Err(Error::Malformed("gguf: bad array elem type")),
                    };
                    let total = (len as usize).checked_mul(elem_size).ok_or(
                        Error::Overflow("gguf: array byte total overflow"),
                    )?;
                    let end = self.pos.checked_add(total).ok_or(
                        Error::Overflow("gguf: array end overflow"),
                    )?;
                    if end > self.buf.len() {
                        return Err(Error::Truncated("gguf: array past end of buffer"));
                    }
                    self.pos = end;
                }
                let data = self.buf[start..self.pos].to_owned();
                GgufValue::Array(GgufArray {
                    elem_type,
                    len,
                    data,
                })
            }
        })
    }
}

// =========================================================================
//                                  TESTS
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Hand-build a minimal GGUF file (one F32 tensor, one Q4_0 tensor,
    /// a couple metadata entries) and verify the parser reconstructs
    /// it exactly. Acts as a spec-compliance test without needing a
    /// real model file checked into the repo.
    #[test]
    fn parser_roundtrips_synthetic_file() {
        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        let tensor_count: u64 = 2;
        let metadata_count: u64 = 3;
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&metadata_count.to_le_bytes());

        // Metadata 1: general.architecture = "llama" (string)
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::STRING as u32).to_le_bytes());
        write_string(&mut buf, "llama");

        // Metadata 2: llama.embedding_length = 288 (u32)
        write_string(&mut buf, "llama.embedding_length");
        buf.extend_from_slice(&(GgufValueType::UINT32 as u32).to_le_bytes());
        buf.extend_from_slice(&288u32.to_le_bytes());

        // Metadata 3: tokenizer.ggml.scores = [1.0, 2.0] (f32 array)
        write_string(&mut buf, "tokenizer.ggml.scores");
        buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes());
        buf.extend_from_slice(&(GgufValueType::FLOAT32 as u32).to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&2.0f32.to_le_bytes());

        // Tensor info 1: "token_embd.weight" — F32, shape [4, 32], offset 0
        write_string(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // Tensor info 2: "blk.0.attn_q.weight" — Q4_0, shape [32, 32], offset ?
        let tensor1_bytes = 4 * 32 * 4; // 512 bytes
        write_string(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
        buf.extend_from_slice(&(tensor1_bytes as u64).to_le_bytes());

        // Pad to alignment (32 by default).
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_base = buf.len();

        // Tensor 1 bytes: 4*32 f32 = 128 floats = 512 bytes
        for i in 0..128u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }
        // Tensor 2 bytes: 32*32 = 1024 elements, Q4_0 = 32 blocks * 18 = 576 bytes
        for _ in 0..576 {
            buf.push(0xAA);
        }

        // Parse.
        let f = GgufFile::parse(&buf).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(f.tensors.len(), 2);
        assert_eq!(f.metadata.len(), 3);

        // Metadata
        assert_eq!(f.metadata_str("general.architecture").unwrap(), "llama");
        assert_eq!(f.metadata_u32("llama.embedding_length").unwrap(), 288);
        match f.metadata.get("tokenizer.ggml.scores").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.elem_type, GgufValueType::FLOAT32);
                assert_eq!(arr.len, 2);
                assert_eq!(arr.data.len(), 8);
            }
            _ => panic!("expected array"),
        }

        // Tensor 1
        let t1 = f.tensor("token_embd.weight").unwrap();
        assert_eq!(t1.dtype, GgufTensorType::F32);
        assert_eq!(t1.shape, vec![4u64, 32u64]);
        assert_eq!(t1.bytes.len(), 512);
        assert_eq!(t1.bytes.as_ptr() as usize, buf[data_base..].as_ptr() as usize);

        // Tensor 2
        let t2 = f.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(t2.dtype, GgufTensorType::Q4_0);
        assert_eq!(t2.shape, vec![32u64, 32u64]);
        assert_eq!(t2.bytes.len(), 576);
        assert_eq!(t2.numel(), 1024);
        // Each byte we wrote was 0xAA — verify slice points at our bytes.
        assert!(t2.bytes.iter().all(|&b| b == 0xAA));
    }

    /// Rejects a malformed file (bad magic, truncated, unsupported version).
    #[test]
    fn parser_rejects_bad_magic() {
        let buf = b"XXXXversion".to_vec();
        let err = GgufFile::parse(&buf).unwrap_err();
        assert!(matches!(err, Error::Malformed(s) if s.contains("bad magic")));
    }

    #[test]
    fn parser_rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let err = GgufFile::parse(&buf).unwrap_err();
        assert!(matches!(err, Error::Malformed(s) if s.contains("unsupported version")));
    }

    #[test]
    fn parser_rejects_truncated_header() {
        let buf = b"GGUF\x03\x00\x00".to_vec();
        let err = GgufFile::parse(&buf).unwrap_err();
        assert!(matches!(err, Error::Truncated(s) if s.contains("past end")));
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
}
