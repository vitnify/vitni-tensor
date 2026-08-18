//! Quantized weights view — the GGUF-backed analog of the F32
//! `Weights` struct.
//!
//! `Weights` (in `weights.rs`) is `&[f32]` everywhere. `QuantizedWeights`
//! is the same logical Llama layout but each projection matrix is a
//! byte slice that can be Q4_0 or Q8_0, and each norm vector stays
//! F32. Forward passes that consume this struct call
//! `linear_q4_0` / `linear_q8_0` instead of plain matmul.
//!
//! ## Tensor name mapping (GGUF Llama convention)
//!
//! GGUF llama files use these tensor names (verified against
//! llama.cpp's `convert.py` and confirmed loading TinyLlama 1.1B
//! Q4_0 from HuggingFace):
//!
//! | Field                     | GGUF tensor name              |
//! |---------------------------|-------------------------------|
//! | token_embedding_table     | `token_embd.weight`           |
//! | rms_att_weight[L]         | `blk.{L}.attn_norm.weight`    |
//! | wq[L]                     | `blk.{L}.attn_q.weight`       |
//! | wk[L]                     | `blk.{L}.attn_k.weight`       |
//! | wv[L]                     | `blk.{L}.attn_v.weight`       |
//! | wo[L]                     | `blk.{L}.attn_output.weight`  |
//! | rms_ffn_weight[L]         | `blk.{L}.ffn_norm.weight`     |
//! | w1[L] (gate)              | `blk.{L}.ffn_gate.weight`     |
//! | w2[L] (down)              | `blk.{L}.ffn_down.weight`     |
//! | w3[L] (up)                | `blk.{L}.ffn_up.weight`       |
//! | rms_final_weight          | `output_norm.weight`          |
//! | wcls (lm_head)            | `output.weight` (optional)    |
//!
//! When `output.weight` is absent the model ties lm_head to
//! `token_embd.weight` (shared embedding); the loader records this
//! by leaving `wcls = wq[0].bytes` as a placeholder and Config's
//! `shared_weights = true` tells the forward pass to use the
//! embedding table instead.

use alloc::format;
use alloc::vec::Vec;

use super::config::Config;
use super::gguf::{GgufFile, GgufTensor, GgufTensorType, GgufValue};
use crate::error::{Error, Result};

/// One projection matrix in a Llama block. Carries its dtype so the
/// forward pass can branch between `linear_q4_0` and plain `matmul`.
#[derive(Debug, Clone)]
pub struct QuantTensor<'a> {
    /// `[out_feat, in_feat]` (GGML stores most weight matrices this way).
    pub shape: Vec<u64>,
    pub dtype: GgufTensorType,
    pub bytes: &'a [u8],
}

impl<'a> From<&GgufTensor<'a>> for QuantTensor<'a> {
    fn from(t: &GgufTensor<'a>) -> Self {
        Self {
            shape: t.shape.clone(),
            dtype: t.dtype,
            bytes: t.bytes,
        }
    }
}

/// Per-layer block of weights — mirrors the per-layer fields in
/// `Weights` but uses `QuantTensor` for matrices and `&[f32]` for
/// the (always-F32) RMSNorm vectors.
#[derive(Debug, Clone)]
pub struct QuantLayer<'a> {
    /// Attention RMSNorm weight — `[dim]`, always F32.
    pub rms_att_weight: &'a [f32],
    /// Query projection — `[dim, n_heads * head_size]`.
    pub wq: QuantTensor<'a>,
    /// Key projection — `[dim, n_kv_heads * head_size]`.
    pub wk: QuantTensor<'a>,
    /// Value projection — `[dim, n_kv_heads * head_size]`.
    pub wv: QuantTensor<'a>,
    /// Output projection — `[n_heads * head_size, dim]`.
    pub wo: QuantTensor<'a>,
    /// FFN RMSNorm weight — `[dim]`, always F32.
    pub rms_ffn_weight: &'a [f32],
    /// FFN gate matrix — `[dim, hidden_dim]`.
    pub w1: QuantTensor<'a>,
    /// FFN down matrix — `[hidden_dim, dim]`.
    pub w2: QuantTensor<'a>,
    /// FFN up matrix — `[dim, hidden_dim]`.
    pub w3: QuantTensor<'a>,
    /// Q/K/V projection biases — present in Qwen2 (`[n_heads*head_size]`,
    /// `[kv_dim]`, `[kv_dim]`, always F32), absent in Llama-family models.
    pub bq: Option<&'a [f32]>,
    pub bk: Option<&'a [f32]>,
    pub bv: Option<&'a [f32]>,
}

/// Whole-model view of GGUF Llama weights. Borrow lifetime `'a`
/// chains back to the underlying file blob.
#[derive(Debug)]
pub struct QuantizedWeights<'a> {
    /// Token embedding table — `[vocab, dim]`. Often F32 or F16 in
    /// GGUF (token-embed is the most cache-sensitive matrix; many
    /// quantization recipes leave it un-quantized).
    pub token_embedding_table: QuantTensor<'a>,
    /// Per-layer blocks, in order.
    pub layers: Vec<QuantLayer<'a>>,
    /// Final RMSNorm — `[dim]`, always F32.
    pub rms_final_weight: &'a [f32],
    /// LM head — `[vocab, dim]`. Present only when the model has a
    /// separate `output.weight`; otherwise the forward pass uses
    /// `token_embedding_table` (Config::shared_weights = true).
    pub wcls: Option<QuantTensor<'a>>,
    /// RoPE base frequency (`{arch}.rope.freq_base`): 10000 for Llama, 1e6 for Qwen2.
    pub rope_theta: f32,
    /// RMSNorm epsilon (`{arch}.attention.layer_norm_rms_epsilon`): 1e-5 Llama, 1e-6 Qwen2.
    pub rms_eps: f32,
    /// True for NeoX split-half RoPE (Qwen2); false for Llama interleaved RoPE.
    pub rope_neox: bool,
}

impl<'a> QuantizedWeights<'a> {
    /// Build a `QuantizedWeights` view from a parsed GGUF file +
    /// the derived `Config`.
    pub fn from_gguf(gguf: &GgufFile<'a>, cfg: &Config) -> Result<Self> {
        let token_embedding_table = QuantTensor::from(get_tensor(gguf, "token_embd.weight")?);

        let rms_final_weight = read_f32_tensor(gguf, "output_norm.weight")?;

        let wcls = match gguf.tensor("output.weight") {
            Some(t) => Some(QuantTensor::from(t)),
            None => None,
        };
        if wcls.is_some() == cfg.shared_weights {
            // Sanity: Config::from_gguf decides shared_weights from
            // presence of output.weight. This is just a redundant
            // check in case caller passed a hand-built Config.
            return Err(Error::Internal(
                "gguf: shared_weights inconsistent with output.weight presence",
            ));
        }

        // Architecture-dependent hyperparameters. Defaults match Llama-2
        // (rope base 10000, rms eps 1e-5, interleaved RoPE); Qwen2 overrides
        // via GGUF metadata (rope base 1e6, rms eps 1e-6, NeoX RoPE + QKV bias).
        let arch = gguf.metadata_str("general.architecture").unwrap_or("llama");
        let read_f32_meta = |key: &str, default: f32| -> f32 {
            match gguf.metadata_value(key) {
                Some(GgufValue::F32(v)) => *v,
                _ => default,
            }
        };
        let rope_theta = read_f32_meta(&format!("{}.rope.freq_base", arch), 10000.0);
        let rms_eps = read_f32_meta(&format!("{}.attention.layer_norm_rms_epsilon", arch), 1e-5);
        let rope_neox = arch == "qwen2";

        let mut layers: Vec<QuantLayer<'a>> = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let blk = QuantLayer {
                rms_att_weight: read_f32_tensor(gguf, &format!("blk.{}.attn_norm.weight", l))?,
                wq: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.attn_q.weight", l))?),
                wk: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.attn_k.weight", l))?),
                wv: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.attn_v.weight", l))?),
                wo: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.attn_output.weight", l))?),
                rms_ffn_weight: read_f32_tensor(gguf, &format!("blk.{}.ffn_norm.weight", l))?,
                w1: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.ffn_gate.weight", l))?),
                w2: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.ffn_down.weight", l))?),
                w3: QuantTensor::from(get_tensor(gguf, &format!("blk.{}.ffn_up.weight", l))?),
                bq: read_f32_tensor_opt(gguf, &format!("blk.{}.attn_q.bias", l))?,
                bk: read_f32_tensor_opt(gguf, &format!("blk.{}.attn_k.bias", l))?,
                bv: read_f32_tensor_opt(gguf, &format!("blk.{}.attn_v.bias", l))?,
            };
            layers.push(blk);
        }

        Ok(Self {
            token_embedding_table,
            layers,
            rms_final_weight,
            wcls,
            rope_theta,
            rms_eps,
            rope_neox,
        })
    }

    /// `(quantized_bytes, total_bytes)` across all matrices. Useful
    /// for printing a load summary like "loaded 4.1 GB, 87 %
    /// quantized".
    pub fn quantization_ratio(&self) -> (u64, u64) {
        let mut quant = 0u64;
        let mut total = 0u64;
        let bump = |t: &QuantTensor<'_>, q: &mut u64, n: &mut u64| {
            *n += t.bytes.len() as u64;
            if matches!(
                t.dtype,
                GgufTensorType::Q4_0 | GgufTensorType::Q8_0
                | GgufTensorType::Q4_K | GgufTensorType::Q6_K
            ) {
                *q += t.bytes.len() as u64;
            }
        };
        bump(&self.token_embedding_table, &mut quant, &mut total);
        for l in &self.layers {
            bump(&l.wq, &mut quant, &mut total);
            bump(&l.wk, &mut quant, &mut total);
            bump(&l.wv, &mut quant, &mut total);
            bump(&l.wo, &mut quant, &mut total);
            bump(&l.w1, &mut quant, &mut total);
            bump(&l.w2, &mut quant, &mut total);
            bump(&l.w3, &mut quant, &mut total);
        }
        if let Some(w) = &self.wcls {
            bump(w, &mut quant, &mut total);
        }
        // RMSNorm vectors are always F32 — count them in `total` only.
        total += self.rms_final_weight.len() as u64 * 4;
        for l in &self.layers {
            total += l.rms_att_weight.len() as u64 * 4;
            total += l.rms_ffn_weight.len() as u64 * 4;
        }
        (quant, total)
    }
}

fn get_tensor<'a, 'b>(gguf: &'b GgufFile<'a>, name: &str) -> Result<&'b GgufTensor<'a>> {
    gguf.tensor(name).ok_or(Error::Internal(
        "gguf: required Llama tensor not found",
    ))
}

/// Read a tensor that MUST be F32 and return a slice view. Errors if
/// dtype is anything else.
fn read_f32_tensor<'a>(gguf: &GgufFile<'a>, name: &str) -> Result<&'a [f32]> {
    let t = get_tensor(gguf, name)?;
    if t.dtype != GgufTensorType::F32 {
        return Err(Error::Internal(
            "gguf: expected F32 norm-weight tensor, got something else",
        ));
    }
    if t.bytes.len() % 4 != 0 {
        return Err(Error::Internal(
            "gguf: F32 tensor byte length not a multiple of 4",
        ));
    }
    // SAFETY: dtype is F32 (checked), so 4-byte alignment is correct
    // and bytes / 4 yields the element count.
    Ok(unsafe {
        core::slice::from_raw_parts(t.bytes.as_ptr() as *const f32, t.bytes.len() / 4)
    })
}

/// Like `read_f32_tensor` but returns `None` if the tensor is absent
/// (used for optional QKV biases: present in Qwen2, absent in Llama).
fn read_f32_tensor_opt<'a>(gguf: &GgufFile<'a>, name: &str) -> Result<Option<&'a [f32]>> {
    match gguf.tensor(name) {
        None => Ok(None),
        Some(t) => {
            if t.dtype != GgufTensorType::F32 {
                return Err(Error::Internal("gguf: expected F32 bias tensor"));
            }
            if t.bytes.len() % 4 != 0 {
                return Err(Error::Internal("gguf: F32 bias byte length not a multiple of 4"));
            }
            // SAFETY: dtype is F32 (checked); 4-byte alignment holds.
            Ok(Some(unsafe {
                core::slice::from_raw_parts(t.bytes.as_ptr() as *const f32, t.bytes.len() / 4)
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gguf::GgufValueType;
    use alloc::string::String;
    use alloc::vec;

    /// Hand-build a minimal Llama-shaped GGUF (2 layers, dim=32,
    /// hidden=64, vocab=8) and verify QuantizedWeights pulls out
    /// each tensor by the expected name.
    #[test]
    fn quant_weights_load_synthetic_llama_gguf() {
        let buf = build_tiny_llama_gguf(2, 32, 64, 8, 4, 4);
        let gguf = GgufFile::parse(&buf).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();

        assert_eq!(cfg.dim, 32);
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.vocab_size, 8);

        let qw = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        assert_eq!(qw.layers.len(), 2);
        // GGUF stores shape in GGML order (innermost/fastest-varying
        // dim first) — for token_embd that's [dim, vocab], not the
        // PyTorch-style [vocab, dim].
        assert_eq!(qw.token_embedding_table.shape, vec![32u64, 8u64]);
        assert_eq!(qw.token_embedding_table.dtype, GgufTensorType::Q4_0);
        assert_eq!(qw.layers[0].wq.shape, vec![32u64, 32u64]);
        assert_eq!(qw.layers[0].wq.dtype, GgufTensorType::Q4_0);
        assert_eq!(qw.rms_final_weight.len(), 32);
        // Synthetic file omits output.weight → shared embedding mode.
        assert!(qw.wcls.is_none());
        assert!(cfg.shared_weights);

        let (q, t) = qw.quantization_ratio();
        assert!(q > 0);
        assert!(t > q); // F32 norms inflate `total` past `quantized`
    }

    /// Build a GGUF that includes a separate output.weight; verify
    /// loader picks it up and shared_weights is false.
    #[test]
    fn quant_weights_with_explicit_output_weight() {
        let mut buf = build_tiny_llama_gguf(1, 32, 64, 8, 4, 4);
        // Re-encode but with an extra tensor — easier path: build
        // from scratch using the helper that knows to include it.
        buf = build_tiny_llama_gguf_with_output(1, 32, 64, 8, 4, 4);
        let gguf = GgufFile::parse(&buf).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        assert!(!cfg.shared_weights);
        let qw = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        assert!(qw.wcls.is_some());
        let wcls = qw.wcls.unwrap();
        // Same GGML reverse-order convention as token_embd above.
        assert_eq!(wcls.shape, vec![32u64, 8u64]);
    }

    // ---------------------------------------------------------------
    // GGUF synthesis helpers — hand-build a llama-shaped GGUF buffer
    // without pulling in llama.cpp's convert.py. These match the tensor
    // names and metadata keys llama.cpp writes for Llama-family models.
    // ---------------------------------------------------------------

    fn build_tiny_llama_gguf(
        n_layers: usize,
        dim: usize,
        hidden_dim: usize,
        vocab: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Vec<u8> {
        build_tiny_llama_gguf_inner(n_layers, dim, hidden_dim, vocab, n_heads, n_kv_heads, false)
    }

    fn build_tiny_llama_gguf_with_output(
        n_layers: usize,
        dim: usize,
        hidden_dim: usize,
        vocab: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Vec<u8> {
        build_tiny_llama_gguf_inner(n_layers, dim, hidden_dim, vocab, n_heads, n_kv_heads, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tiny_llama_gguf_inner(
        n_layers: usize,
        dim: usize,
        hidden_dim: usize,
        vocab: usize,
        n_heads: usize,
        n_kv_heads: usize,
        include_output: bool,
    ) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // ---- Header ----
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());

        // We'll patch tensor_count + metadata_count back in after we know.
        let tc_offset = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

        // ---- Metadata ----
        let mut mc = 0u64;
        write_str_kv(&mut buf, "general.architecture", "llama");
        mc += 1;
        write_u32_kv(&mut buf, "llama.embedding_length", dim as u32);
        mc += 1;
        write_u32_kv(&mut buf, "llama.feed_forward_length", hidden_dim as u32);
        mc += 1;
        write_u32_kv(&mut buf, "llama.block_count", n_layers as u32);
        mc += 1;
        write_u32_kv(&mut buf, "llama.attention.head_count", n_heads as u32);
        mc += 1;
        write_u32_kv(&mut buf, "llama.attention.head_count_kv", n_kv_heads as u32);
        mc += 1;
        write_u32_kv(&mut buf, "llama.context_length", 256u32);
        mc += 1;
        write_u32_kv(&mut buf, "llama.vocab_size", vocab as u32);
        mc += 1;

        // ---- Tensor infos ----
        // Track (name, shape, dtype, offset). Offsets are computed AFTER
        // we know the tensor data section base.
        let mut tinfos: Vec<(String, Vec<u64>, u32, u64)> = Vec::new();
        let mut tc = 0u64;
        let mut cur_off: u64 = 0;

        // Push a tensor info + reserve its bytes.
        let push = |tinfos: &mut Vec<_>, cur_off: &mut u64, name: &str, shape: Vec<u64>, dtype: u32| {
            let numel: u64 = shape.iter().product();
            let nbytes: u64 = match dtype {
                0 => numel * 4,                       // F32
                2 => (numel / 32) * 18,               // Q4_0
                _ => panic!("test helper doesn't support dtype {}", dtype),
            };
            tinfos.push((name.into(), shape, dtype, *cur_off));
            *cur_off += nbytes;
        };

        push(&mut tinfos, &mut cur_off, "token_embd.weight", vec![dim as u64, vocab as u64], 2);
        tc += 1;
        for l in 0..n_layers {
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.attn_norm.weight", l), vec![dim as u64], 0);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.attn_q.weight",     l), vec![dim as u64, (n_heads * (dim / n_heads)) as u64], 2);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.attn_k.weight",     l), vec![dim as u64, (n_kv_heads * (dim / n_heads)) as u64], 2);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.attn_v.weight",     l), vec![dim as u64, (n_kv_heads * (dim / n_heads)) as u64], 2);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.attn_output.weight",l), vec![(n_heads * (dim / n_heads)) as u64, dim as u64], 2);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.ffn_norm.weight",   l), vec![dim as u64], 0);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.ffn_gate.weight",   l), vec![dim as u64, hidden_dim as u64], 2);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.ffn_down.weight",   l), vec![hidden_dim as u64, dim as u64], 2);
            push(&mut tinfos, &mut cur_off, &format!("blk.{}.ffn_up.weight",     l), vec![dim as u64, hidden_dim as u64], 2);
            tc += 9;
        }
        push(&mut tinfos, &mut cur_off, "output_norm.weight", vec![dim as u64], 0);
        tc += 1;
        if include_output {
            push(&mut tinfos, &mut cur_off, "output.weight", vec![dim as u64, vocab as u64], 2);
            tc += 1;
        }

        // Patch tensor_count + metadata_count.
        buf[tc_offset..tc_offset + 8].copy_from_slice(&tc.to_le_bytes());
        buf[tc_offset + 8..tc_offset + 16].copy_from_slice(&mc.to_le_bytes());

        // Write tensor infos.
        for (name, shape, dtype, offset) in &tinfos {
            write_string(&mut buf, name);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        // Pad to alignment.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Write tensor data (just zeros — we're testing parsing not content).
        let data_len: u64 = tinfos
            .iter()
            .map(|(_, shape, dtype, _)| {
                let numel: u64 = shape.iter().product();
                match dtype {
                    0 => numel * 4,
                    2 => (numel / 32) * 18,
                    _ => 0,
                }
            })
            .sum();
        buf.extend(core::iter::repeat(0u8).take(data_len as usize));

        buf
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    fn write_str_kv(buf: &mut Vec<u8>, key: &str, val: &str) {
        write_string(buf, key);
        buf.extend_from_slice(&(GgufValueType::STRING as u32).to_le_bytes());
        write_string(buf, val);
    }
    fn write_u32_kv(buf: &mut Vec<u8>, key: &str, val: u32) {
        write_string(buf, key);
        buf.extend_from_slice(&(GgufValueType::UINT32 as u32).to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }
}
