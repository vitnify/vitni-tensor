//! Model-agnostic byte-level BPE tokenizer (GPT-2 family: `tokenizer.ggml.model = "gpt2"`),
//! reading vocab + merges straight from the GGUF — covers Qwen2.5 / Llama-3.
//!
//! Pipeline (GPT-2 / Qwen2 style):
//!   pretokenize -> map each byte to its GPT-2 byte-unicode char -> merge by merge-rank
//!   -> look the merged pieces up in the vocab.
//! The pretokenizer here is a pragmatic ASCII-aware scanner matching the Qwen2/GPT-4
//! word boundaries for typical prompts; full unicode-property-regex parity is a follow-on.
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use crate::error::{Error, Result};
use crate::model::gguf::{GgufFile, GgufValue, GgufValueType};

pub struct GgufBpeTokenizer {
    vocab: BTreeMap<Vec<u8>, u32>,                 // token string (byte-unicode) -> id
    id_to_tok: Vec<Vec<u8>>,                       // id -> token string
    merge_rank: BTreeMap<(Vec<u8>, Vec<u8>), u32>, // (left,right) byte-unicode -> rank
    byte_encoder: [char; 256],                     // GPT-2 bytes_to_unicode
}

fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut used = [false; 256];
    for &(lo, hi) in &[(b'!' as u32, b'~' as u32), (0xA1u32, 0xACu32), (0xAEu32, 0xFFu32)] {
        for b in lo..=hi { table[b as usize] = char::from_u32(b).unwrap(); used[b as usize] = true; }
    }
    let mut n = 0u32;
    for b in 0u32..256 {
        if !used[b as usize] { table[b as usize] = char::from_u32(256 + n).unwrap(); n += 1; }
    }
    table
}

impl GgufBpeTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let toks = read_str_array(gguf, "tokenizer.ggml.tokens")?;
        let merges = read_str_array(gguf, "tokenizer.ggml.merges")?;
        let mut vocab = BTreeMap::new();
        for (i, t) in toks.iter().enumerate() { vocab.entry(t.clone()).or_insert(i as u32); }
        let mut merge_rank = BTreeMap::new();
        for (rank, m) in merges.iter().enumerate() {
            // each merge is "A B" (space-separated byte-unicode pieces)
            if let Some(sp) = m.iter().position(|&c| c == b' ') {
                let (a, b) = (m[..sp].to_vec(), m[sp + 1..].to_vec());
                merge_rank.entry((a, b)).or_insert(rank as u32);
            }
        }
        Ok(Self { vocab, id_to_tok: toks, merge_rank, byte_encoder: bytes_to_unicode() })
    }

    pub fn vocab_size(&self) -> usize { self.id_to_tok.len() }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pretokenize(text) {
            // bytes -> byte-unicode symbols
            let mut symbols: Vec<Vec<u8>> = Vec::new();
            for &b in piece.as_bytes() {
                let mut s = [0u8; 4];
                symbols.push(self.byte_encoder[b as usize].encode_utf8(&mut s).as_bytes().to_vec());
            }
            self.bpe(&mut symbols);
            for sym in &symbols {
                if let Some(&id) = self.vocab.get(sym) { out.push(id); }
                // (a well-formed gpt2 vocab always resolves single byte-unicode chars)
            }
        }
        out
    }

    fn bpe(&self, symbols: &mut Vec<Vec<u8>>) {
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&r) = self.merge_rank.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if r < best_rank { best_rank = r; best_i = Some(i); }
                }
            }
            match best_i {
                Some(i) => {
                    let mut merged = symbols[i].clone();
                    merged.extend_from_slice(&symbols[i + 1]);
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        // reverse byte-unicode: map each token's chars back to bytes
        let rev: BTreeMap<char, u8> = (0u16..256).map(|b| (self.byte_encoder[b as usize], b as u8)).collect();
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_tok.get(id as usize) {
                if let Ok(s) = core::str::from_utf8(tok) {
                    for ch in s.chars() { if let Some(&b) = rev.get(&ch) { bytes.push(b); } }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Pragmatic Qwen2/GPT-4-style pretokenizer for ASCII-ish text. Splits into: contractions,
/// (optional single non-space/non-alnum or space) + letter-run, digit-runs, (optional space)
/// + punct-run, and whitespace runs.
fn pretokenize(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let n = b.len();
    let is_letter = |c: u8| c.is_ascii_alphabetic() || c >= 0x80; // treat non-ASCII as letter
    let is_digit = |c: u8| c.is_ascii_digit();
    let is_space = |c: u8| c == b' ' || c == b'\t' || c == b'\n' || c == b'\r';
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        let start = i;
        let c = b[i];
        // contraction: 'x
        if c == b'\'' && i + 1 < n {
            let two = &text[i..(i + 2).min(n)];
            let low = two.to_ascii_lowercase();
            if matches!(low.as_str(), "'s" | "'t" | "'m" | "'d") { out.push(text[i..i+2].into()); i += 2; continue; }
            if i + 3 <= n {
                let three = text[i..i+3].to_ascii_lowercase();
                if matches!(three.as_str(), "'re" | "'ve" | "'ll") { out.push(text[i..i+3].into()); i += 3; continue; }
            }
        }
        // optional single leading space, then letters
        if (c == b' ' && i + 1 < n && is_letter(b[i + 1])) || is_letter(c) {
            if c == b' ' { i += 1; }
            while i < n && is_letter(b[i]) { i += 1; }
            out.push(text[start..i].into()); continue;
        }
        // digits (each run; llama.cpp Qwen groups per-digit but runs are fine for round-trip via vocab)
        if is_digit(c) || (c == b' ' && i + 1 < n && is_digit(b[i + 1])) {
            if c == b' ' { i += 1; }
            while i < n && is_digit(b[i]) { i += 1; }
            out.push(text[start..i].into()); continue;
        }
        // optional leading space, then punctuation run (non-space, non-alnum)
        if (c == b' ' && i + 1 < n && !is_space(b[i + 1]) && !is_letter(b[i + 1]) && !is_digit(b[i + 1]))
            || (!is_space(c) && !is_letter(c) && !is_digit(c)) {
            if c == b' ' { i += 1; }
            while i < n && !is_space(b[i]) && !is_letter(b[i]) && !is_digit(b[i]) { i += 1; }
            out.push(text[start..i].into()); continue;
        }
        // whitespace run
        if is_space(c) {
            while i < n && is_space(b[i]) { i += 1; }
            out.push(text[start..i].into()); continue;
        }
        i += 1; // safety
    }
    out
}

fn read_str_array(gguf: &GgufFile, key: &str) -> Result<Vec<Vec<u8>>> {
    match gguf.metadata_value(key) {
        Some(GgufValue::Array(arr)) if arr.elem_type == GgufValueType::STRING => {
            let d = &arr.data;
            let mut out = Vec::with_capacity(arr.len as usize);
            let mut p = 0usize;
            for _ in 0..arr.len {
                if p + 8 > d.len() { return Err(Error::InvalidShape("gguf str array truncated")); }
                let ln = u64::from_le_bytes(d[p..p + 8].try_into().unwrap()) as usize;
                p += 8;
                if p + ln > d.len() { return Err(Error::InvalidShape("gguf str element past end")); }
                out.push(d[p..p + ln].to_vec());
                p += ln;
            }
            Ok(out)
        }
        _ => Err(Error::InvalidShape("gguf: expected a string array")),
    }
}
