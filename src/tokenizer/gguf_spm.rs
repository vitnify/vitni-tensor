//! Model-agnostic SentencePiece tokenizer that reads the vocab + scores straight
//! from a GGUF file (`tokenizer.ggml.tokens` / `tokenizer.ggml.scores`). Any
//! Llama-family model (`tokenizer.ggml.model = "llama"`) tokenizes with ITS OWN
//! vocab — no external tokenizer, no cross-model mismatch. Covers TinyLlama,
//! Mistral, Llama-2, etc. (SentencePiece/unigram). Byte-level BPE models
//! (Llama-3, Qwen; `model = "gpt2"`) are a separate follow-on.
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use crate::error::{Error, Result};
use crate::model::gguf::{GgufFile, GgufValue, GgufValueType};

const MARK: &[u8] = &[0xE2, 0x96, 0x81]; // "▁" U+2581 (SentencePiece space marker)

pub struct GgufSpmTokenizer {
    vocab: Vec<Vec<u8>>,               // token id -> UTF-8 bytes of the token string
    scores: Vec<f32>,
    by_bytes: BTreeMap<Vec<u8>, u32>,  // reverse index
    byte_fallback: [Option<u32>; 256], // byte value -> "<0xHH>" token id
    max_len: usize,
}

impl GgufSpmTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let vocab = read_str_array(gguf, "tokenizer.ggml.tokens")?;
        let scores = read_f32_array(gguf, "tokenizer.ggml.scores")?;
        if vocab.len() != scores.len() {
            return Err(Error::InvalidShape("gguf tokenizer: tokens/scores length mismatch"));
        }
        let mut by_bytes = BTreeMap::new();
        let mut byte_fallback: [Option<u32>; 256] = [None; 256];
        let mut max_len = 0usize;
        for (i, t) in vocab.iter().enumerate() {
            by_bytes.entry(t.clone()).or_insert(i as u32);
            if t.len() > max_len { max_len = t.len(); }
            if t.len() == 6 && t[0] == b'<' && t[1] == b'0' && t[2] == b'x' && t[5] == b'>' {
                if let Some(b) = hex2(t[3], t[4]) { byte_fallback[b as usize] = Some(i as u32); }
            }
        }
        Ok(Self { vocab, scores, by_bytes, byte_fallback, max_len })
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    /// SentencePiece encode: prepend a space + replace spaces with ▁, split into
    /// UTF-8 chars (byte-fallback for out-of-vocab), then greedily merge the
    /// highest-scoring adjacent pair that forms a vocab token.
    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        // normalize: leading ▁, and every space -> ▁
        let mut norm: Vec<u8> = Vec::with_capacity(text.len() + 3);
        norm.extend_from_slice(MARK);
        for ch in text.chars() {
            if ch == ' ' { norm.extend_from_slice(MARK); }
            else { let mut b = [0u8; 4]; norm.extend_from_slice(ch.encode_utf8(&mut b).as_bytes()); }
        }
        let s = core::str::from_utf8(&norm).unwrap_or("");

        let mut tokens: Vec<u32> = Vec::new();
        if bos { tokens.push(1); } // <s>
        let start = tokens.len();
        for ch in s.chars() {
            let mut b = [0u8; 4];
            let cb = ch.encode_utf8(&mut b).as_bytes();
            if let Some(&id) = self.by_bytes.get(cb) {
                tokens.push(id);
            } else {
                for &byte in cb {
                    if let Some(id) = self.byte_fallback[byte as usize] { tokens.push(id); }
                }
            }
        }

        let mut tmp: Vec<u8> = Vec::with_capacity(self.max_len * 2);
        loop {
            let mut best = f32::NEG_INFINITY;
            let mut best_id: Option<u32> = None;
            let mut best_idx: Option<usize> = None;
            for i in start..tokens.len().saturating_sub(1) {
                tmp.clear();
                tmp.extend_from_slice(&self.vocab[tokens[i] as usize]);
                tmp.extend_from_slice(&self.vocab[tokens[i + 1] as usize]);
                if let Some(&id) = self.by_bytes.get(&tmp) {
                    if self.scores[id as usize] > best {
                        best = self.scores[id as usize];
                        best_id = Some(id);
                        best_idx = Some(i);
                    }
                }
            }
            match (best_id, best_idx) {
                (Some(id), Some(i)) => { tokens[i] = id; tokens.remove(i + 1); }
                _ => break,
            }
        }
        tokens
    }

    /// Decode token IDs to text: map each token's bytes back, "<0xHH>" -> raw byte,
    /// ▁ -> space, and drop the single synthetic leading space.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut out: Vec<u8> = Vec::new();
        for &t in tokens {
            let v = &self.vocab[t as usize];
            if v.len() == 6 && v[0] == b'<' && v[1] == b'0' && v[2] == b'x' && v[5] == b'>' {
                if let Some(b) = hex2(v[3], v[4]) { out.push(b); continue; }
            }
            out.extend_from_slice(v);
        }
        let s = String::from_utf8_lossy(&out).replace('\u{2581}', " ");
        match s.strip_prefix(' ') {
            Some(rest) => String::from(rest),
            None => s,
        }
    }
}

fn hex2(a: u8, b: u8) -> Option<u8> {
    let h = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    Some((h(a)? << 4) | h(b)?)
}

fn read_str_array(gguf: &GgufFile, key: &str) -> Result<Vec<Vec<u8>>> {
    match gguf.metadata_value(key) {
        Some(GgufValue::Array(arr)) if arr.elem_type == GgufValueType::STRING => {
            let d = &arr.data;
            let mut out = Vec::with_capacity(arr.len as usize);
            let mut p = 0usize;
            for _ in 0..arr.len {
                if p + 8 > d.len() { return Err(Error::InvalidShape("gguf str array truncated")); }
                let n = u64::from_le_bytes(d[p..p + 8].try_into().unwrap()) as usize;
                p += 8;
                if p + n > d.len() { return Err(Error::InvalidShape("gguf str element past end")); }
                out.push(d[p..p + n].to_vec());
                p += n;
            }
            Ok(out)
        }
        _ => Err(Error::InvalidShape("gguf: tokenizer.ggml.tokens missing or not a string array")),
    }
}

fn read_f32_array(gguf: &GgufFile, key: &str) -> Result<Vec<f32>> {
    match gguf.metadata_value(key) {
        Some(GgufValue::Array(arr)) if arr.elem_type == GgufValueType::FLOAT32 => {
            let d = &arr.data;
            let mut out = Vec::with_capacity(arr.len as usize);
            for i in 0..arr.len as usize {
                let o = i * 4;
                if o + 4 > d.len() { return Err(Error::InvalidShape("gguf f32 array truncated")); }
                out.push(f32::from_bits(u32::from_le_bytes(d[o..o + 4].try_into().unwrap())));
            }
            Ok(out)
        }
        _ => Err(Error::InvalidShape("gguf: tokenizer.ggml.scores missing or not an f32 array")),
    }
}
