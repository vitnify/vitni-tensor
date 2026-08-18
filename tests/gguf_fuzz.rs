//! Adversarial GGUF parser tests. Feed the loader deliberately
//! malformed byte streams — bad magic, truncated headers, absurd
//! counts/dimensions, lengths that run past EOF — and assert every
//! one comes back as `Err` without panicking, aborting, or hanging.
//!
//! The GGUF file is attacker-controllable (downloaded checkpoints,
//! untrusted CAS blobs), so a corrupt one must never take the process
//! down. These exercise the untrusted-input paths in `model/gguf.rs`.
//!
//! Run: cargo test --test gguf_fuzz

use vitni_tensor::model::gguf::{GgufFile, GgufValueType};

/// GGUF magic bytes.
const MAGIC: &[u8; 4] = b"GGUF";

/// Append a GGUF string: `u64` little-endian length followed by bytes.
fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Build a well-formed header: magic + version + tensor_count + meta_count.
fn header(version: u32, tensor_count: u64, meta_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&meta_count.to_le_bytes());
    buf
}

/// Append one tensor-info record: name + n_dims + dims + ggml_type + offset.
fn push_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], ggml_type: u32, offset: u64) {
    push_str(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&ggml_type.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

// ---------------------------------------------------------------
// Header / magic / version
// ---------------------------------------------------------------

#[test]
fn empty_input() {
    assert!(GgufFile::parse(&[]).is_err());
}

#[test]
fn truncated_magic() {
    assert!(GgufFile::parse(b"GG").is_err());
    assert!(GgufFile::parse(b"GGU").is_err());
}

#[test]
fn bad_magic() {
    let mut buf = b"XXXX".to_vec();
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn truncated_after_magic() {
    // Magic present, but no version field follows.
    assert!(GgufFile::parse(MAGIC).is_err());
}

#[test]
fn unsupported_versions() {
    for v in [0u32, 1, 4, 99, u32::MAX] {
        let buf = header(v, 0, 0);
        assert!(GgufFile::parse(&buf).is_err(), "version {v} should be rejected");
    }
}

#[test]
fn truncated_tensor_count() {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&[0u8, 0, 0]); // only 3 of the 8 tensor_count bytes
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn truncated_metadata_count() {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    buf.extend_from_slice(&[0u8, 0, 0, 0]); // partial metadata_count
    assert!(GgufFile::parse(&buf).is_err());
}

// ---------------------------------------------------------------
// Absurd counts — must not pre-allocate huge buffers or spin
// ---------------------------------------------------------------

#[test]
fn tensor_count_max_does_not_allocate_or_hang() {
    // u64::MAX tensors claimed, but no tensor-info bytes follow. The
    // capacity reservation must be bounded and the read loop must bail.
    let buf = header(3, u64::MAX, 0);
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn metadata_count_max_does_not_hang() {
    let buf = header(3, 0, u64::MAX);
    assert!(GgufFile::parse(&buf).is_err());
}

// ---------------------------------------------------------------
// Metadata keys / values
// ---------------------------------------------------------------

#[test]
fn metadata_key_truncated() {
    // Claims one metadata pair, but the key length runs past EOF.
    let mut buf = header(3, 0, 1);
    buf.extend_from_slice(&64u64.to_le_bytes()); // key len = 64, no bytes follow
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn metadata_string_over_cap() {
    // Key length exceeds the 1 MiB sanity cap — must error before the
    // allocation, not attempt a 2 MiB read.
    let mut buf = header(3, 0, 1);
    buf.extend_from_slice(&(2u64 << 20).to_le_bytes()); // 2 MiB
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn unknown_metadata_value_type() {
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "k");
    buf.extend_from_slice(&999u32.to_le_bytes()); // not a valid value-type tag
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn nested_array_rejected() {
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "arr");
    buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes()); // value is array
    buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes()); // elem is array (nested)
    buf.extend_from_slice(&1u64.to_le_bytes());
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn array_len_overflow_fixed_elem() {
    // len * sizeof(elem) overflows usize.
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "arr");
    buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes());
    buf.extend_from_slice(&(GgufValueType::UINT32 as u32).to_le_bytes()); // 4-byte elems
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // len = u64::MAX
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn array_fixed_elem_past_eof() {
    // Claims four u32 (16 bytes), none of which are present.
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "arr");
    buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes());
    buf.extend_from_slice(&(GgufValueType::UINT32 as u32).to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn array_of_strings_len_max() {
    // u64::MAX string elements — the per-element read loop must hit EOF
    // and bail rather than spin to completion.
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "arr");
    buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes());
    buf.extend_from_slice(&(GgufValueType::STRING as u32).to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn array_string_len_past_eof() {
    // One string element whose declared length runs ~1 TiB past EOF.
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "arr");
    buf.extend_from_slice(&(GgufValueType::ARRAY as u32).to_le_bytes());
    buf.extend_from_slice(&(GgufValueType::STRING as u32).to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // one string
    buf.extend_from_slice(&(1u64 << 40).to_le_bytes()); // its length
    assert!(GgufFile::parse(&buf).is_err());
}

// ---------------------------------------------------------------
// Tensor infos — dims / dtype / offset
// ---------------------------------------------------------------

#[test]
fn tensor_rank_too_high() {
    let mut buf = header(3, 1, 0);
    push_str(&mut buf, "t");
    buf.extend_from_slice(&9u32.to_le_bytes()); // n_dims = 9 > 8 cap
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_dims_product_overflows() {
    // Two dims whose product wraps u64.
    let mut buf = header(3, 1, 0);
    push_tensor_info(&mut buf, "t", &[u64::MAX, 2], 0 /* F32 */, 0);
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_dims_truncated() {
    // n_dims claims 4 dims; only one dim's bytes are present.
    let mut buf = header(3, 1, 0);
    push_str(&mut buf, "t");
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&8u64.to_le_bytes()); // 1 of 4 dims
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_numel_not_block_aligned() {
    // Q4_0 requires numel % 32 == 0; 33 is not. Trailing bytes are
    // present so the failure is the block check, not EOF.
    let mut buf = header(3, 1, 0);
    push_tensor_info(&mut buf, "t", &[33], 2 /* Q4_0 */, 0);
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&[0u8; 4096]);
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_unsupported_ggml_type() {
    // ggml type 20 has no known block size — must error, not panic.
    let mut buf = header(3, 1, 0);
    push_tensor_info(&mut buf, "t", &[32], 20 /* Other */, 0);
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&[0u8; 4096]);
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_offset_past_eof() {
    // Valid F32 tensor, but the offset points far past the blob.
    let mut buf = header(3, 1, 0);
    push_tensor_info(&mut buf, "t", &[4], 0 /* F32 */, u64::MAX / 2);
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&[0u8; 64]);
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_offset_overflow_add() {
    // offset = u64::MAX so base + offset overflows usize.
    let mut buf = header(3, 1, 0);
    push_tensor_info(&mut buf, "t", &[32], 0 /* F32 */, u64::MAX);
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&[0u8; 128]);
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn tensor_data_truncated() {
    // F32 [1024] = 4096 bytes declared, but no data section is present.
    let mut buf = header(3, 1, 0);
    push_tensor_info(&mut buf, "t", &[1024], 0 /* F32 */, 0);
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    assert!(GgufFile::parse(&buf).is_err());
}

#[test]
fn alignment_zero_no_divide_by_zero() {
    // general.alignment = 0 would divide-by-zero in the padding calc.
    let mut buf = header(3, 0, 1);
    push_str(&mut buf, "general.alignment");
    buf.extend_from_slice(&(GgufValueType::UINT32 as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // alignment = 0
    assert!(GgufFile::parse(&buf).is_err());
}

// ---------------------------------------------------------------
// Positive control — the hardening must not reject valid input
// ---------------------------------------------------------------

#[test]
fn minimal_valid_file_still_parses() {
    // Header with zero tensors and zero metadata is a valid (if empty)
    // GGUF; confirms the guards didn't break the happy path.
    let buf = header(3, 0, 0);
    let f = GgufFile::parse(&buf).expect("empty-but-valid gguf should parse");
    assert_eq!(f.version, 3);
    assert_eq!(f.tensors.len(), 0);
    assert!(f.metadata.is_empty());
}
