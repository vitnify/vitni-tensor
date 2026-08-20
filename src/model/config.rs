//! Llama2 config — 7 i32 header from the karpathy llama2.c binary format.

use alloc::string::ToString;

use crate::error::{Error, Result};

/// Model dimensions. Mirrors karpathy's `llama2.c` `Config` struct.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Hidden dimension of the residual stream.
    pub dim: usize,
    /// FFN intermediate dimension (typically 4 * dim for Llama).
    pub hidden_dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads for Q.
    pub n_heads: usize,
    /// Number of key/value heads. Equals `n_heads` for full MHA,
    /// less for grouped-query attention.
    pub n_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length the KV cache must accommodate.
    pub seq_len: usize,
    /// If true, the lm_head matrix aliases the token embedding table
    /// (smaller-model convention, e.g. stories15M).
    pub shared_weights: bool,
}

impl Config {
    /// Parse the 28-byte (7 × i32) header at the start of a llama2.c
    /// weights blob.
    ///
    /// llama2.c convention: a negative `vocab_size` in the header
    /// signals `shared_weights = false` (the lm_head is a separate
    /// matrix); positive means shared.
    pub fn from_header(blob: &[u8]) -> Result<Self> {
        if blob.len() < 28 {
            return Err(Error::InvalidShape("config header < 28 bytes"));
        }
        let read = |off: usize| -> i32 {
            let b: [u8; 4] = blob[off..off + 4].try_into().unwrap();
            i32::from_le_bytes(b)
        };
        let dim = read(0) as usize;
        let hidden_dim = read(4) as usize;
        let n_layers = read(8) as usize;
        let n_heads = read(12) as usize;
        let n_kv_heads = read(16) as usize;
        let raw_vocab = read(20);
        let seq_len = read(24) as usize;
        let shared_weights = raw_vocab > 0;
        let vocab_size = raw_vocab.unsigned_abs() as usize;
        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights,
        })
    }

    /// Per-head dimension. `dim / n_heads`.
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// KV-side dimension. Equals `dim` for full MHA, less for GQA.
    pub fn kv_dim(&self) -> usize {
        (self.dim * self.n_kv_heads) / self.n_heads
    }

    /// GQA multiplier: how many Q heads share each KV head.
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// stories15M-shaped synthetic config for tests.
    pub const fn stories15m() -> Self {
        Self {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
            shared_weights: true,
        }
    }

    /// Mistral 7B v0.1 dims. Architecture identical to Llama2 except
    /// for GQA (n_kv_heads < n_heads), which `forward::step` already
    /// handles via `kv_mul = n_heads / n_kv_heads`. Set
    /// `shared_weights = false` because Mistral has a separate
    /// `lm_head` matrix (not tied to the input embedding table).
    ///
    /// Verification: with these dims plumbed through, the EXISTING
    /// `forward::step` function correctly drives a Mistral forward
    /// pass — no per-architecture branch required. this demonstrates
    /// this via `tests/mistral_synthetic.rs`.
    pub const fn mistral_7b_v01() -> Self {
        Self {
            dim: 4096,
            hidden_dim: 14336,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8, // GQA: 4 Q heads per KV head
            vocab_size: 32000,
            seq_len: 4096, // truncated from 32768 for test-friendly KV cache size
            shared_weights: false,
        }
    }

    /// Gemma 2B dims. Architecture differs from Llama2 in three small
    /// ways requiring a separate `forward::gemma::step`:
    ///   1. RMSNorm scales by `(1.0 + weight)` instead of `weight`.
    ///   2. The token-embedding input is multiplied by `sqrt(dim)`.
    ///   3. FFN uses GeLU(gate) * up (vs Llama's SiLU(gate) * up).
    /// See `forward::gemma`.
    pub const fn gemma_2b() -> Self {
        Self {
            dim: 2048,
            hidden_dim: 16384,
            n_layers: 18,
            n_heads: 8,
            n_kv_heads: 1, // multi-query attention (extreme GQA)
            vocab_size: 256000,
            seq_len: 8192,
            shared_weights: true, // Gemma ties lm_head to embedding table
        }
    }

    /// Construct a `Config` from a parsed GGUF file's metadata.
    ///
    /// The mapping uses the keys that llama.cpp's `convert.py` writes
    /// for Llama-family models (Llama 2, Mistral, TinyLlama, etc.):
    ///   `{arch}.embedding_length`        → dim
    ///   `{arch}.feed_forward_length`     → hidden_dim
    ///   `{arch}.block_count`             → n_layers
    ///   `{arch}.attention.head_count`    → n_heads
    ///   `{arch}.attention.head_count_kv` → n_kv_heads (optional; defaults to n_heads)
    ///   `{arch}.context_length`          → seq_len
    ///   `tokenizer.ggml.tokens` array len → vocab_size
    ///
    /// where `{arch}` is `general.architecture` (commonly "llama" for
    /// Llama 2 / Mistral / TinyLlama; "gemma" for Gemma).
    ///
    /// `shared_weights` is inferred from the presence of an `output.weight`
    /// tensor — if absent, the lm_head aliases `token_embd.weight`
    /// (Llama 1, stories15M convention); if present, they're separate
    /// (Mistral 7B convention).
    pub fn from_gguf(gguf: &super::gguf::GgufFile<'_>) -> Result<Self> {
        let arch = gguf
            .metadata_str("general.architecture")
            .map_err(|_| Error::Internal(
                "gguf: missing 'general.architecture' metadata",
            ))?
            .to_string();

        let prefix = |suffix: &str| -> alloc::string::String {
            let mut s = arch.clone();
            s.push('.');
            s.push_str(suffix);
            s
        };

        let read_u32_loose = |key: &str| -> Result<usize> {
            match gguf.metadata.get(key) {
                Some(super::gguf::GgufValue::U32(v)) => Ok(*v as usize),
                Some(super::gguf::GgufValue::U64(v)) => Ok(*v as usize),
                Some(super::gguf::GgufValue::I32(v)) if *v >= 0 => Ok(*v as usize),
                Some(super::gguf::GgufValue::I64(v)) if *v >= 0 => Ok(*v as usize),
                _ => Err(Error::Internal(
                    "gguf: numeric metadata missing or wrong type",
                )),
            }
        };

        let dim = read_u32_loose(&prefix("embedding_length"))?;
        let hidden_dim = read_u32_loose(&prefix("feed_forward_length"))?;
        let n_layers = read_u32_loose(&prefix("block_count"))?;
        let n_heads = read_u32_loose(&prefix("attention.head_count"))?;
        let n_kv_heads = read_u32_loose(&prefix("attention.head_count_kv"))
            .unwrap_or(n_heads);
        let seq_len = read_u32_loose(&prefix("context_length"))?;

        // Vocab size: prefer the array length over the metadata u32 because
        // some tokenizer configs lie and the array is authoritative.
        let vocab_size = match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(super::gguf::GgufValue::Array(arr)) => arr.len as usize,
            _ => read_u32_loose(&prefix("vocab_size"))?,
        };

        // Shared-weights inference: if output.weight is present the
        // model has a distinct lm_head. If absent it shares the
        // input embedding table (Llama 1 / stories15M).
        let shared_weights = gguf.tensor("output.weight").is_none();

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_synthetic_header() {
        // Build a 28-byte header matching stories15M and parse it.
        let mut blob = alloc::vec::Vec::with_capacity(28);
        for &v in &[288i32, 768, 6, 6, 6, 32000, 256] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let cfg = Config::from_header(&blob).unwrap();
        assert_eq!(cfg.dim, 288);
        assert_eq!(cfg.hidden_dim, 768);
        assert_eq!(cfg.n_layers, 6);
        assert_eq!(cfg.n_heads, 6);
        assert_eq!(cfg.n_kv_heads, 6);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.seq_len, 256);
        assert!(cfg.shared_weights);
        assert_eq!(cfg.head_size(), 48);
        assert_eq!(cfg.kv_dim(), 288);
    }

    #[test]
    fn shared_weights_flag() {
        let mut blob = alloc::vec::Vec::with_capacity(28);
        // Negative vocab_size signals NOT shared.
        for &v in &[288i32, 768, 6, 6, 6, -32000, 256] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let cfg = Config::from_header(&blob).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert!(!cfg.shared_weights);
    }

    #[test]
    fn short_blob_errors() {
        let blob = alloc::vec![0u8; 20];
        assert!(Config::from_header(&blob).is_err());
    }
}
