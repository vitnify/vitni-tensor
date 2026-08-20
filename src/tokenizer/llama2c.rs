//! BPE tokenizer for karpathy's llama2.c binary tokenizer format.
//!
//! Wire format (matches what karpathy's `tokenizer.bin` and
//! `the reference` consume):
//!
//!   u32  max_token_length
//!   for i in 0..vocab_size:
//!     f32  score
//!     u32  len
//!     [u8; len] bytes
//!
//! Encoding is greedy BPE: start with one ID per byte, then
//! repeatedly merge the highest-scoring adjacent pair, until no
//! merge is possible. O(N²) per step but fast on real prompts
//! (~10-100 bytes).
//!
//! Ported from `userspace/the reference/src/main.rs::Tokenizer`
//! so behavior is bit-identical — same prompt → same token IDs
//! → same cert binding.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// BPE tokenizer parsed from a karpathy llama2.c tokenizer.bin blob.
pub struct Llama2cTokenizer {
    /// vocab[i] = the byte sequence for token id `i`.
    pub vocab: Vec<Vec<u8>>,
    /// Per-token merge score (higher = preferred earlier in BPE).
    pub scores: Vec<f32>,
    /// Longest token byte-length in the vocab (advisory; used to
    /// size the merge-attempt buffer).
    pub max_token_length: usize,
    /// Reverse index: bytes → vocab id. Built once at parse time
    /// so encode doesn't go quadratic over scan + lookup.
    by_bytes: BTreeMap<Vec<u8>, u32>,
}

impl Llama2cTokenizer {
    /// Parse a llama2.c tokenizer.bin blob.
    ///
    /// `vocab_size` must come from the matching model config (the
    /// blob doesn't self-describe its vocab count). Mismatch → blob
    /// is truncated / over-read; we return `InvalidShape` rather
    /// than panic.
    pub fn from_blob(blob: &[u8], vocab_size: usize) -> Result<Self> {
        if blob.len() < 4 {
            return Err(Error::InvalidShape("tokenizer blob < 4 bytes"));
        }
        let mut p = 0usize;
        let read_u32 = |off: usize, blob: &[u8]| -> u32 {
            u32::from_le_bytes([blob[off], blob[off + 1], blob[off + 2], blob[off + 3]])
        };
        let max_token_length = read_u32(p, blob) as usize;
        p += 4;

        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        let mut scores: Vec<f32> = Vec::with_capacity(vocab_size);
        let mut by_bytes: BTreeMap<Vec<u8>, u32> = BTreeMap::new();

        for i in 0..vocab_size {
            if p + 8 > blob.len() {
                return Err(Error::InvalidShape(
                    "tokenizer blob truncated mid-entry header",
                ));
            }
            let score = f32::from_bits(read_u32(p, blob));
            p += 4;
            let len = read_u32(p, blob) as usize;
            p += 4;
            if p + len > blob.len() {
                return Err(Error::InvalidShape("tokenizer blob truncated mid-token"));
            }
            let bytes = blob[p..p + len].to_vec();
            // First-occurrence wins on collisions (matches karpathy's
            // `entry().or_insert()` semantics).
            by_bytes.entry(bytes.clone()).or_insert(i as u32);
            vocab.push(bytes);
            scores.push(score);
            p += len;
        }

        Ok(Self {
            vocab,
            scores,
            max_token_length,
            by_bytes,
        })
    }

    /// Greedy BPE encode: start with one ID per byte, then merge
    /// the highest-scoring adjacent pair until no merge applies.
    ///
    /// `bos = true` prepends BOS (token id 1, by Llama2 convention).
    /// For continuation, pass `false`.
    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();
        if bos {
            tokens.push(1); // <s>
        }
        for ch in text.bytes() {
            let single = vec![ch];
            if let Some(&idx) = self.by_bytes.get(&single) {
                tokens.push(idx);
            }
            // If a single byte isn't in the vocab, llama2.c's tokenizer
            // typically has a <0xHH> fallback. The fallback path is
            // model-specific; stories15M's vocab covers all 256 single
            // bytes so we don't hit this path in practice.
        }

        let mut tmp: Vec<u8> = Vec::with_capacity(self.max_token_length * 2);
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id: Option<u32> = None;
            let mut best_idx: Option<usize> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                tmp.clear();
                tmp.extend_from_slice(&self.vocab[tokens[i] as usize]);
                tmp.extend_from_slice(&self.vocab[tokens[i + 1] as usize]);
                if let Some(&merged) = self.by_bytes.get(&tmp) {
                    if self.scores[merged as usize] > best_score {
                        best_score = self.scores[merged as usize];
                        best_id = Some(merged);
                        best_idx = Some(i);
                    }
                }
            }
            match (best_id, best_idx) {
                (Some(id), Some(idx)) => {
                    tokens[idx] = id;
                    tokens.remove(idx + 1);
                }
                _ => break,
            }
        }
        tokens
    }

    /// Decode a single token, handling the two llama2.c quirks:
    ///   1. After BOS (id 1), strip a leading space from the next
    ///      token's bytes (artifact of how the Llama tokenizer
    ///      attaches a space to most word-starting tokens).
    ///   2. Raw-byte tokens of the form `<0xHH>` decode to the
    ///      single raw byte HH (used for arbitrary bytes outside
    ///      the BPE merge set).
    pub fn decode(&self, prev_token: u32, token: u32) -> Vec<u8> {
        let mut piece = self.vocab[token as usize].clone();
        if prev_token == 1 && piece.first() == Some(&b' ') {
            piece.remove(0);
        }
        if piece.len() >= 6 && &piece[..3] == b"<0x" && piece.last() == Some(&b'>') {
            if let Ok(s) = core::str::from_utf8(&piece[3..piece.len() - 1]) {
                if let Ok(byte_val) = u8::from_str_radix(s, 16) {
                    return vec![byte_val];
                }
            }
        }
        piece
    }

    /// Decode a sequence of token IDs (lossy: bytes that aren't
    /// valid UTF-8 are replaced with `?`). For lossless output use
    /// `decode_bytes` and inspect the raw bytes.
    pub fn decode_string(&self, prev_token: u32, tokens: &[u32]) -> String {
        let bytes = self.decode_bytes(prev_token, tokens);
        let mut out = String::with_capacity(bytes.len());
        for chunk in bytes.utf8_chunks() {
            out.push_str(chunk.valid());
            if !chunk.invalid().is_empty() {
                out.push('?');
            }
        }
        out
    }

    /// Decode a sequence of token IDs to raw bytes. `prev_token` is
    /// the token immediately preceding `tokens[0]` (use 0 for "no
    /// previous" — the BOS-space stripping only triggers when
    /// `prev_token == 1`).
    pub fn decode_bytes(&self, mut prev: u32, tokens: &[u32]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for &t in tokens {
            out.extend_from_slice(&self.decode(prev, t));
            prev = t;
        }
        out
    }
}

// `utf8_chunks` is on `[u8]` via the `Utf8Chunks` iterator. Using
// the stable `from_utf8_lossy` would pull `Cow` and add a dep —
// keep it minimal with our own loop.
//
// Allowed-dead-code because callers typically use `decode_bytes`
// directly and handle UTF-8 themselves. `decode_string` exists for
// completeness; the helper machinery isn't called from elsewhere
// in the lib but is public surface for downstream code.
#[allow(dead_code)]
trait Utf8ChunksExt {
    fn utf8_chunks(&self) -> Utf8Chunks<'_>;
}
#[allow(dead_code)]
impl Utf8ChunksExt for [u8] {
    fn utf8_chunks(&self) -> Utf8Chunks<'_> {
        Utf8Chunks { remaining: self }
    }
}
#[allow(dead_code)]
struct Utf8Chunks<'a> {
    remaining: &'a [u8],
}
#[allow(dead_code)]
struct Chunk<'a> {
    valid: &'a str,
    invalid: &'a [u8],
}
#[allow(dead_code)]
impl<'a> Chunk<'a> {
    fn valid(&self) -> &'a str {
        self.valid
    }
    fn invalid(&self) -> &'a [u8] {
        self.invalid
    }
}
impl<'a> Iterator for Utf8Chunks<'a> {
    type Item = Chunk<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            return None;
        }
        match core::str::from_utf8(self.remaining) {
            Ok(s) => {
                let result = Chunk {
                    valid: s,
                    invalid: &[],
                };
                self.remaining = &[];
                Some(result)
            }
            Err(e) => {
                let valid_end = e.valid_up_to();
                let valid = core::str::from_utf8(&self.remaining[..valid_end]).unwrap();
                let invalid_len = e.error_len().unwrap_or(self.remaining.len() - valid_end);
                let invalid = &self.remaining[valid_end..valid_end + invalid_len];
                let after = valid_end + invalid_len;
                let chunk = Chunk { valid, invalid };
                self.remaining = &self.remaining[after..];
                Some(chunk)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic tokenizer blob for unit tests. Vocab:
    ///   id 0: "<unk>" (special)
    ///   id 1: "<s>"   (BOS)
    ///   id 2: "</s>"  (EOS)
    ///   id 3..=258: one per byte 0..=255 (256 single-byte tokens)
    ///   id 259: "hi"  (merge for "h" + "i")
    fn build_test_blob() -> (Vec<u8>, usize) {
        let vocab_size = 260;
        let max_token_length = 4u32;
        let mut blob = Vec::new();
        blob.extend_from_slice(&max_token_length.to_le_bytes());
        let push = |b: &mut Vec<u8>, score: f32, bytes: &[u8]| {
            b.extend_from_slice(&score.to_bits().to_le_bytes());
            b.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            b.extend_from_slice(bytes);
        };
        push(&mut blob, -100.0, b"<unk>");
        push(&mut blob, -100.0, b"<s>");
        push(&mut blob, -100.0, b"</s>");
        for byte in 0u8..=255 {
            push(&mut blob, -10.0, &[byte]);
        }
        // Merge token "hi" with high score so encode picks it.
        push(&mut blob, 5.0, b"hi");
        (blob, vocab_size)
    }

    #[test]
    fn parse_blob_roundtrips() {
        let (blob, vocab) = build_test_blob();
        let tok = Llama2cTokenizer::from_blob(&blob, vocab).unwrap();
        assert_eq!(tok.vocab.len(), vocab);
        assert_eq!(tok.vocab[1], b"<s>");
        assert_eq!(tok.vocab[259], b"hi");
        assert!(tok.scores[259] > tok.scores[3]);
    }

    #[test]
    fn truncated_blob_errors() {
        let (mut blob, vocab) = build_test_blob();
        blob.truncate(blob.len() - 5);
        assert!(Llama2cTokenizer::from_blob(&blob, vocab).is_err());
    }

    #[test]
    fn encode_merges_pairs_by_score() {
        let (blob, vocab) = build_test_blob();
        let tok = Llama2cTokenizer::from_blob(&blob, vocab).unwrap();
        let ids = tok.encode("hi", true);
        // Expected: BOS=1, then merged "hi"=259 (instead of "h" + "i").
        assert_eq!(ids, alloc::vec![1, 259]);
    }

    #[test]
    fn encode_no_merge_falls_back_to_bytes() {
        let (blob, vocab) = build_test_blob();
        let tok = Llama2cTokenizer::from_blob(&blob, vocab).unwrap();
        let ids = tok.encode("ab", false); // no "ab" merge defined
        assert_eq!(ids, alloc::vec![3 + b'a' as u32, 3 + b'b' as u32]);
    }

    #[test]
    fn decode_recovers_bytes() {
        let (blob, vocab) = build_test_blob();
        let tok = Llama2cTokenizer::from_blob(&blob, vocab).unwrap();
        let ids = alloc::vec![3 + b'h' as u32, 3 + b'i' as u32];
        assert_eq!(tok.decode_bytes(0, &ids), b"hi".to_vec());
        assert_eq!(tok.decode_string(0, &ids), "hi");
    }

    #[test]
    fn decode_strips_leading_space_after_bos() {
        // Synthetic: add a token " hello" so the BOS-space strip
        // path can be exercised.
        let mut blob = Vec::new();
        blob.extend_from_slice(&4u32.to_le_bytes());
        let push = |b: &mut Vec<u8>, score: f32, bytes: &[u8]| {
            b.extend_from_slice(&score.to_bits().to_le_bytes());
            b.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            b.extend_from_slice(bytes);
        };
        push(&mut blob, -100.0, b"<unk>");
        push(&mut blob, -100.0, b"<s>");
        push(&mut blob, -100.0, b" hello");
        let tok = Llama2cTokenizer::from_blob(&blob, 3).unwrap();
        // Decoding token 2 (" hello") with prev=BOS strips the space.
        let bytes = tok.decode(/* prev = BOS */ 1, 2);
        assert_eq!(bytes, b"hello".to_vec());
        // With prev != BOS, the space stays.
        let bytes = tok.decode(0, 2);
        assert_eq!(bytes, b" hello".to_vec());
    }

    #[test]
    fn decode_raw_byte_token() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&7u32.to_le_bytes());
        let push = |b: &mut Vec<u8>, score: f32, bytes: &[u8]| {
            b.extend_from_slice(&score.to_bits().to_le_bytes());
            b.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            b.extend_from_slice(bytes);
        };
        push(&mut blob, -100.0, b"<unk>");
        push(&mut blob, -100.0, b"<s>");
        push(&mut blob, -10.0, b"<0x0A>"); // raw newline
        let tok = Llama2cTokenizer::from_blob(&blob, 3).unwrap();
        assert_eq!(tok.decode(0, 2), b"\n".to_vec());
    }
}
