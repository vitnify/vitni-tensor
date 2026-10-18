//! CUDA (NVIDIA GPU) inference backend for `vitni-tensor`.
//!
//! `CudaForward` runs the full quantized Llama/Mistral forward on the GPU using
//! the bit-exact kernels in `../vt_cuda.cu` (proven against the CPU in
//! `kernels/cuda_forward_conformance.cu`). `run_cuda` drives it through the
//! engine's `inference::run_quantized_with_forward` seam, so the GPU issues a
//! certificate IDENTICAL to the CPU's — the logits are bit-for-bit identical.
//!
//! The whole model is resident on the GPU: weights + KV cache + activation
//! scratch are allocated once. A decode step launches the forward kernels on the
//! default stream (which serializes them, giving the Metal backend's barrier
//! ordering) and reads back the logits. Only the tiny RoPE cos/sin table is
//! computed per step — on the host with the SAME `libm` the CPU engine uses, so
//! the rotation is identical.

use std::os::raw::{c_char, c_int, c_uint};
use vitni_tensor::cert::NullSink;
use vitni_tensor::model::config::Config;
use vitni_tensor::model::inference::{self, Outcome, Request, RunError};
use vitni_tensor::model::quant_weights::{QuantTensor, QuantizedWeights};
use vitni_tensor::{Error as TError, Shape, Tensor};

#[repr(C)]
struct VtCfg {
    dim: c_int,
    hidden: c_int,
    n_layers: c_int,
    n_heads: c_int,
    head_size: c_int,
    kv_dim: c_int,
    kv_mul: c_int,
    vocab: c_int,
    max_seq: c_int,
    rope_neox: c_int,
    regime: c_int,
    rms_eps: f32,
    rope_theta: f32,
}

#[repr(C)]
struct VtCuda {
    _private: [u8; 0],
}

extern "C" {
    fn vt_cuda_new(cfg: *const VtCfg) -> *mut VtCuda;
    fn vt_cuda_set_wcls(h: *mut VtCuda, b: *const u8, len: usize, q6k: c_int);
    fn vt_cuda_set_rms_final(h: *mut VtCuda, w: *const f32);
    #[allow(clippy::too_many_arguments)]
    fn vt_cuda_set_layer(
        h: *mut VtCuda, l: c_int, rms_att: *const f32, rms_ffn: *const f32,
        wq: *const u8, wq_len: usize, wq_q6k: c_int,
        wk: *const u8, wk_len: usize, wk_q6k: c_int,
        wv: *const u8, wv_len: usize, wv_q6k: c_int,
        wo: *const u8, wo_len: usize, wo_q6k: c_int,
        w1: *const u8, w1_len: usize, w1_q6k: c_int,
        w2: *const u8, w2_len: usize, w2_q6k: c_int,
        w3: *const u8, w3_len: usize, w3_q6k: c_int,
        bq: *const f32, bk: *const f32, bv: *const f32,
    );
    fn vt_cuda_finalize(h: *mut VtCuda) -> c_int;
    fn vt_cuda_reset(h: *mut VtCuda);
    fn vt_cuda_step(h: *mut VtCuda, x_embed: *const f32, pos: c_uint, cosc: *const f32, sinc: *const f32, out_logits: *mut f32) -> c_int;
    fn vt_cuda_device_name() -> *const c_char;
    fn vt_cuda_test_linear(kind: c_int, k: c_int, m: c_int, x: *const f32, w: *const u8, wlen: usize, out: *mut f32, iters: c_int) -> c_int;
    fn vt_cuda_test_linear_int(k: c_int, m: c_int, x: *const f32, w: *const u8, wlen: usize, out: *mut f32, mode: c_int, iters: c_int) -> c_int;
}

/// TIER-2 (regime-3) integer Q4_K linear: out[M]=W.x[K] via int8-activation integer dot.
/// mode 0 = plain integer loop, 1 = __dp4a. iters repeats the dot (for timing).
pub fn test_linear_int(k: usize, m: usize, x: &[f32], w: &[u8], mode: i32, iters: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    let rc = unsafe { vt_cuda_test_linear_int(k as c_int, m as c_int, x.as_ptr(), w.as_ptr(), w.len(), out.as_mut_ptr(), mode, iters) };
    assert_eq!(rc, 0, "vt_cuda_test_linear_int rc={rc}");
    out
}

/// Run one linear of `kind` (0=Q4_K,1=Q6_K,2=Q8_0,3=Q5_0) on the GPU: out[M]=W[M,K].x[K].
pub fn test_linear(kind_code: i32, k: usize, m: usize, x: &[f32], w: &[u8]) -> Vec<f32> {
    test_linear_n(kind_code, k, m, x, w, 1)
}
pub fn test_linear_n(kind_code: i32, k: usize, m: usize, x: &[f32], w: &[u8], iters: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    let rc = unsafe { vt_cuda_test_linear(kind_code, k as c_int, m as c_int, x.as_ptr(), w.as_ptr(), w.len(), out.as_mut_ptr(), iters) };
    assert_eq!(rc, 0, "vt_cuda_test_linear rc={rc}");
    out
}

#[derive(Debug)]
pub enum CudaError {
    Alloc(String),
    Unsupported(String),
    Device(i32),
    Engine(TError),
}
impl core::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CudaError::Alloc(s) => write!(f, "device allocation/upload failed: {s}"),
            CudaError::Unsupported(s) => write!(f, "unsupported model: {s}"),
            CudaError::Device(e) => write!(f, "CUDA error {e}"),
            CudaError::Engine(e) => write!(f, "engine: {e:?}"),
        }
    }
}
impl std::error::Error for CudaError {}
impl From<TError> for CudaError {
    fn from(e: TError) -> Self { CudaError::Engine(e) }
}

/// Human-readable GPU name, e.g. "Tesla T4 (CC 7.5)".
pub fn device_name() -> String {
    unsafe {
        let p = vt_cuda_device_name();
        if p.is_null() { return "unknown".into(); }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Weight block type the kernels handle: Q4_K (0), Q6_K (1), Q8_0 (2). Q4_K_M
/// GGUFs are a Q4_K/Q6_K mix, sometimes with a Q8_0 head; anything else is unported.
fn kind(t: &QuantTensor, name: &str) -> Result<c_int, CudaError> {
    let dt = format!("{:?}", t.dtype);
    match dt.as_str() {
        "Q4_K" => Ok(0),
        "Q6_K" => Ok(1),
        "Q8_0" => Ok(2),
        "Q5_0" => Ok(3),
        other => Err(CudaError::Unsupported(format!("{name} is {other}; only Q4_K/Q6_K/Q8_0/Q5_0 are ported"))),
    }
}

/// A quantized Llama/Mistral/Qwen2 model resident on the NVIDIA GPU. Borrows the
/// embedding-table bytes (the per-token lookup is dequantized on the host, which
/// matches the CPU exactly for any embed quant type).
pub struct CudaForward<'w> {
    h: *mut VtCuda,
    head_size: usize,
    rope_theta: f32,
    vocab: usize,
    dim: usize,
    embed_bytes: &'w [u8],
    embed_dtype: String,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl<'w> CudaForward<'w> {
    /// regime-2 (shipped): float Q4_K + fixed_tree Q6_K.
    pub fn new(cfg: &Config, w: &'w QuantizedWeights<'w>) -> Result<CudaForward<'w>, CudaError> {
        Self::new_regime(cfg, w, 2)
    }
    /// regime 2 (shipped) or 3 (integer-activation + running-sum k-quant linears, matching
    /// inference::run_quantized_regime3). regime-3 is unpublished.
    pub fn new_regime(cfg: &Config, w: &'w QuantizedWeights<'w>, regime: i32) -> Result<CudaForward<'w>, CudaError> {
        // Supported: interleaved (Llama/Mistral) and NeoX (Qwen2) RoPE, optional
        // QKV bias (Qwen2). The embedding lookup is done on the host for any dtype.
        let head_size = cfg.head_size();
        let q_out = cfg.n_heads * head_size;
        let dim = cfg.dim;
        let kv_dim = cfg.kv_dim();
        let hidden = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        // KV cache length. Cap to bound GPU memory (a 32B's KV at full ctx is huge);
        // VITNI_MAX_SEQ overrides (set it small on a 24GB card). Must exceed the
        // longest sequence actually run.
        let cap = std::env::var("VITNI_MAX_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        let max_seq = cfg.seq_len.min(cap);

        let c = VtCfg {
            dim: dim as c_int,
            hidden: hidden as c_int,
            n_layers: cfg.n_layers as c_int,
            n_heads: cfg.n_heads as c_int,
            head_size: head_size as c_int,
            kv_dim: kv_dim as c_int,
            kv_mul: cfg.kv_mul() as c_int,
            vocab: vocab as c_int,
            max_seq: max_seq as c_int,
            rope_neox: if w.rope_neox { 1 } else { 0 },
            regime: regime as c_int,
            rms_eps: w.rms_eps,
            rope_theta: w.rope_theta,
        };

        let h = unsafe { vt_cuda_new(&c) };
        if h.is_null() { return Err(CudaError::Alloc("vt_cuda_new".into())); }

        let e = &w.token_embedding_table;
        unsafe {
            match &w.wcls {
                Some(c) => vt_cuda_set_wcls(h, c.bytes.as_ptr(), c.bytes.len(), kind(c, "output")?),
                // tied: classifier == embedding (a dim%256 Q4_K table in practice)
                None => vt_cuda_set_wcls(h, e.bytes.as_ptr(), e.bytes.len(), kind(e, "tied-embedding")?),
            }
            vt_cuda_set_rms_final(h, w.rms_final_weight.as_ptr());
            let np = core::ptr::null::<f32>();
            for (i, l) in w.layers.iter().enumerate() {
                vt_cuda_set_layer(
                    h, i as c_int, l.rms_att_weight.as_ptr(), l.rms_ffn_weight.as_ptr(),
                    l.wq.bytes.as_ptr(), l.wq.bytes.len(), kind(&l.wq, "wq")?,
                    l.wk.bytes.as_ptr(), l.wk.bytes.len(), kind(&l.wk, "wk")?,
                    l.wv.bytes.as_ptr(), l.wv.bytes.len(), kind(&l.wv, "wv")?,
                    l.wo.bytes.as_ptr(), l.wo.bytes.len(), kind(&l.wo, "wo")?,
                    l.w1.bytes.as_ptr(), l.w1.bytes.len(), kind(&l.w1, "w1")?,
                    l.w2.bytes.as_ptr(), l.w2.bytes.len(), kind(&l.w2, "w2")?,
                    l.w3.bytes.as_ptr(), l.w3.bytes.len(), kind(&l.w3, "w3")?,
                    l.bq.map_or(np, |b| b.as_ptr()),
                    l.bk.map_or(np, |b| b.as_ptr()),
                    l.bv.map_or(np, |b| b.as_ptr()),
                );
            }
            if vt_cuda_finalize(h) != 0 {
                return Err(CudaError::Alloc("weight upload / scratch allocation (out of GPU memory?)".into()));
            }
        }

        Ok(CudaForward {
            h,
            head_size,
            rope_theta: w.rope_theta,
            vocab,
            dim,
            embed_bytes: e.bytes,
            embed_dtype: format!("{:?}", e.dtype),
            cos: vec![0.0; head_size / 2],
            sin: vec![0.0; head_size / 2],
        })
    }

    /// Dequantize one row of the embedding table on the host, matching the CPU
    /// `embedding_row` exactly (bit-for-bit) for any supported quant type.
    fn embed_row(&self, token: usize) -> Result<Vec<f32>, CudaError> {
        use vitni_tensor::ops::quant::{dequantize_q4_0, dequantize_q4_k, dequantize_q5_0, dequantize_q6_k, dequantize_q8_0};
        let dim = self.dim;
        let b = self.embed_bytes;
        let slice = |rb: usize| -> &[u8] { &b[token * rb..(token + 1) * rb] };
        let row = match self.embed_dtype.as_str() {
            "F32" => {
                let off = token * dim * 4;
                (0..dim).map(|i| f32::from_le_bytes([b[off + i * 4], b[off + i * 4 + 1], b[off + i * 4 + 2], b[off + i * 4 + 3]])).collect()
            }
            "Q4_0" => dequantize_q4_0(slice((dim / 32) * 18))?,
            "Q8_0" => dequantize_q8_0(slice((dim / 32) * 34))?,
            "Q5_0" => dequantize_q5_0(slice((dim / 32) * 22))?,
            "Q4_K" => dequantize_q4_k(slice((dim / 256) * 144))?,
            "Q6_K" => dequantize_q6_k(slice((dim / 256) * 210))?,
            other => return Err(CudaError::Unsupported(format!("embedding dtype {other}"))),
        };
        Ok(row)
    }

    /// Clear the KV cache to start a fresh sequence (certificate-neutral).
    pub fn reset(&mut self) {
        unsafe { vt_cuda_reset(self.h) };
    }

    /// RoPE cos/sin for `pos`, computed with the same `libm` as the CPU engine.
    fn rope_cache(&mut self, pos: usize) {
        let half = self.head_size / 2;
        for j in 0..half {
            let freq = 1.0f32 / libm::powf(self.rope_theta, (2 * j) as f32 / self.head_size as f32);
            let val = pos as f32 * freq;
            self.cos[j] = libm::cosf(val);
            self.sin[j] = libm::sinf(val);
        }
    }

    /// Logits for `token` at sequence position `pos`, advancing the GPU KV cache.
    pub fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        self.rope_cache(pos);
        let x = self.embed_row(token as usize).expect("embedding lookup");
        let mut logits = vec![0.0f32; self.vocab];
        let rc = unsafe {
            vt_cuda_step(self.h, x.as_ptr(), pos as c_uint, self.cos.as_ptr(), self.sin.as_ptr(), logits.as_mut_ptr())
        };
        if rc != 0 {
            // step is on the hot path and returns Vec; a device error here is fatal.
            panic!("vt_cuda_step failed: CUDA error {rc}");
        }
        logits
    }
}

impl<'w> Drop for CudaForward<'w> {
    fn drop(&mut self) {
        // Process-lifetime model; the OS reclaims device memory on exit. We keep
        // the handle for the process (no vt_cuda_free) to avoid a double-free of
        // the many device pointers if Drop races teardown.
    }
}

/// Run a full generation on the NVIDIA GPU with a freshly-loaded model and issue
/// a certificate identical to `inference::run_quantized` on CPU.
pub fn run_cuda(
    cfg: &Config,
    weights: &QuantizedWeights,
    weights_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome, CudaError> {
    let mut cf = CudaForward::new(cfg, weights)?;
    run_cuda_with(&mut cf, cfg, weights_hash, req)
}

/// Regime-3 GPU forward: integer-activation + running-sum k-quant linears. Its certificate
/// must equal `inference::run_quantized_regime3` on CPU (same regime-3 logits).
pub fn run_cuda_regime3(
    cfg: &Config,
    weights: &QuantizedWeights,
    weights_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome, CudaError> {
    let mut cf = CudaForward::new_regime(cfg, weights, 3)?;
    run_cuda_with(&mut cf, cfg, weights_hash, req)
}

// (run_cuda_with below is generic over the forward's borrow.)

/// Like `run_cuda` but reuses an already-loaded `CudaForward` (resets the KV
/// cache first). This is what a server does: load the model once, serve many
/// requests. The certificate is independent of prior KV state.
pub fn run_cuda_with(
    cf: &mut CudaForward,
    cfg: &Config,
    weights_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome, CudaError> {
    // cf: &mut CudaForward<'_> — the embedding borrow is elided here.
    cf.reset();
    let vocab = cf.vocab;
    let (outcome, _id) = inference::run_quantized_with_forward(
        cfg, weights_hash, req, &mut NullSink, 0,
        |cur, pos| Tensor::from_f32(cf.step(cur, pos), Shape::new(&[vocab])?),
    )
    .map_err(|e| match e {
        RunError::Inference(err) => CudaError::Engine(err),
        RunError::Sink(_) => unreachable!("NullSink is infallible"),
    })?;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vitni_tensor::model::gguf::GgufFile;

    /// Localize a GPU/CPU divergence: compare CudaForward::step against the CPU
    /// forward_quantized::step for the first few positions and report the first
    /// differing logit (index + magnitude). An immediate, large diff points at a
    /// quant-matmul kernel; a small/late diff points at rope/bias/attention.
    #[test]
    #[ignore = "debug; set VITNI_GGUF_7B"]
    fn cuda_debug_divergence() {
        use vitni_tensor::model::{forward::RunState, forward_quantized};
        let path = std::env::var("VITNI_GGUF_7B").expect("set VITNI_GGUF_7B");
        let file = std::fs::File::open(&path).expect("open gguf");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap gguf") };
        let blob: &[u8] = &mmap;
        let gguf = GgufFile::parse(blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        eprintln!("device: {}  rope_neox={}  dim={} head_size={} n_heads={} kv_dim={}",
            device_name(), weights.rope_neox, cfg.dim, cfg.head_size(), cfg.n_heads, cfg.kv_dim());
        let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
        let mut gpu = CudaForward::new(&cfg, &weights).unwrap();
        gpu.reset();
        let mut st = RunState::new(&cfg);
        for (pos, &tok) in prompt.iter().enumerate() {
            let gl = gpu.step(tok, pos);
            let ct = forward_quantized::step(&cfg, &weights, &mut st, tok, pos).unwrap();
            let cb = ct.storage_cpu_bytes().unwrap();
            let cl: &[f32] = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const f32, cb.len() / 4) };
            let mut first = None; let mut maxabs = 0.0f32; let mut ndiff = 0usize;
            for i in 0..gl.len() {
                let d = (gl[i] - cl[i]).abs();
                if gl[i].to_bits() != cl[i].to_bits() { ndiff += 1; if first.is_none() { first = Some(i); } if d > maxabs { maxabs = d; } }
            }
            match first {
                None => eprintln!("pos {pos}: logits BIT-IDENTICAL ({} elems)", gl.len()),
                Some(i) => { eprintln!("pos {pos}: {ndiff} diffs, first @ {i}  gpu={} cpu={}  maxabs={maxabs:e}", gl[i], cl[i]); break; }
            }
        }
    }

    /// Isolate the Q5_0 / Q8_0 kernels: run layer-0 wq and the output tensor through
    /// the GPU linear and compare to the CPU dequant + canonical_dot, element-by-bit.
    #[test]
    #[ignore = "debug; set VITNI_GGUF_7B"]
    fn cuda_isolate_quant_kernels() {
        use vitni_tensor::ops::quant::{canonical_dot_pub, dequantize_q5_0, dequantize_q8_0};
        let path = std::env::var("VITNI_GGUF_7B").expect("set VITNI_GGUF_7B");
        let blob = std::fs::read(&path).expect("read gguf");
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        let dim = cfg.dim;
        // deterministic pseudo-random x in a realistic range
        let x: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.6180339887).fract() - 0.5) * 2.0).collect();

        let cases: &[(&str, &QuantTensor, usize, usize)] = &[
            ("wq (layer0)", &weights.layers[0].wq, cfg.n_heads * cfg.head_size(), dim),
            ("wk (layer0)", &weights.layers[0].wk, cfg.kv_dim(), dim),
            ("output",      weights.wcls.as_ref().unwrap_or(&weights.token_embedding_table), cfg.vocab_size, dim),
        ];
        for (name, t, m, k) in cases {
            let dt = format!("{:?}", t.dtype);
            let (code, rowb, cpu_row): (i32, usize, fn(&[u8]) -> Vec<f32>) = match dt.as_str() {
                "Q5_0" => (3, (k / 32) * 22, |b| dequantize_q5_0(b).unwrap()),
                "Q8_0" => (2, (k / 32) * 34, |b| dequantize_q8_0(b).unwrap()),
                other => { eprintln!("{name}: {other} (skip — covered by Mistral)"); continue; }
            };
            let gpu = test_linear(code, *k, *m, &x, t.bytes);
            let mut first = None; let mut nd = 0usize;
            for o in 0..*m {
                let row = cpu_row(&t.bytes[o * rowb..(o + 1) * rowb]);
                let cpu = canonical_dot_pub(&x, &row, *k);
                if cpu.to_bits() != gpu[o].to_bits() { nd += 1; if first.is_none() { first = Some((o, cpu, gpu[o])); } }
            }
            match first {
                None => eprintln!("{name} [{dt}]: {m} rows BIT-IDENTICAL (GPU==CPU)"),
                Some((o, c, g)) => eprintln!("{name} [{dt}]: {nd}/{m} differ, first @ row {o}  cpu={c} gpu={g}"),
            }
        }
    }

    /// TIER-2 spike: integer-activation Q4_K linear (regime-3). Proves (1) the GPU
    /// integer dot is bit-identical to the CPU integer reference (dot_q4k_q8_block),
    /// (2) a plain loop and a __dp4a path give the IDENTICAL result (integer sums are
    /// associative → order/method-independent → safe for warp reductions & int8 tensor
    /// cores), and (3) __dp4a is faster. Run with VITNI_GGUF_7B set to any Q4_K model.
    #[test]
    #[ignore = "tier-2 spike; set VITNI_GGUF_7B; needs an NVIDIA GPU"]
    fn tier2_integer_q4k() {
        use std::time::Instant;
        use vitni_tensor::ops::quant::{quantize_block_q8, dot_q4k_q8_block, canonical_combine};
        let path = std::env::var("VITNI_GGUF_7B").expect("set VITNI_GGUF_7B");
        let file = std::fs::File::open(&path).expect("open gguf");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap gguf") };
        let blob: &[u8] = &mmap;
        let gguf = GgufFile::parse(blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        eprintln!("device: {}", device_name());

        // pick a Q4_K layer-0 tensor (ffn gate w1 is big → good timing), K = its input dim
        let l0 = &weights.layers[0];
        let cands: &[(&str, &QuantTensor, usize, usize)] = &[
            ("w1", &l0.w1, cfg.hidden_dim, cfg.dim),
            ("wq", &l0.wq, cfg.n_heads * cfg.head_size(), cfg.dim),
        ];
        let (name, t, m, k) = *cands.iter().find(|(_, t, _, _)| format!("{:?}", t.dtype) == "Q4_K")
            .expect("no Q4_K tensor found in layer 0");
        eprintln!("layer-0 {name}: Q4_K, M={m} K={k} nsuper={}", k / 256);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.6180339887).fract() - 0.5) * 2.0).collect();

        // CPU integer reference (regime-3): per block quantize_block_q8 + dot_q4k_q8_block, canonical combine
        let nsuper = k / 256;
        let q8: Vec<_> = (0..nsuper).map(|s| quantize_block_q8(&x[s * 256..(s + 1) * 256])).collect();
        let t0 = Instant::now();
        let mut cpu = vec![0f32; m];
        for row in 0..m {
            let wrow = &t.bytes[row * nsuper * 144..(row + 1) * nsuper * 144];
            let mut parts = vec![0f32; nsuper];
            for s in 0..nsuper { parts[s] = dot_q4k_q8_block(&wrow[s * 144..(s + 1) * 144], &q8[s], 256); }
            cpu[row] = canonical_combine(&mut parts);
        }
        eprintln!("CPU integer reference: {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);

        let gpu_plain = test_linear_int(k, m, &x, t.bytes, 0, 1);  // 1 thread/row, plain loop
        let gpu_dp4a  = test_linear_int(k, m, &x, t.bytes, 1, 1);  // 1 thread/row, __dp4a
        let gpu_warp  = test_linear_int(k, m, &x, t.bytes, 2, 1);  // warp/row, coalesced
        let gpu_wdp4a = test_linear_int(k, m, &x, t.bytes, 3, 1);  // warp/row + __dp4a
        let bit = |a: &[f32], b: &[f32]| (0..a.len()).filter(|&i| a[i].to_bits() == b[i].to_bits()).count();
        eprintln!("GPU plain  == CPU : {}/{} bit-identical", bit(&gpu_plain, &cpu), m);
        eprintln!("GPU __dp4a == CPU : {}/{} bit-identical", bit(&gpu_dp4a, &cpu), m);
        eprintln!("GPU warp   == CPU : {}/{} bit-identical", bit(&gpu_warp, &cpu), m);
        eprintln!("GPU warp+dp4a == CPU : {}/{} bit-identical (4 different reductions, 1 result)", bit(&gpu_wdp4a, &cpu), m);

        // timing: amortize launch over `iters` dots. Also time the FLOAT coop kernel (Tier-1) for comparison.
        let iters = 100;
        let tt = |mode: i32| { let s = Instant::now(); let _ = test_linear_int(k, m, &x, t.bytes, mode, iters); s.elapsed().as_secs_f64() * 1000.0 / iters as f64 };
        let (plain_ms, dp4a_ms, warp_ms, wdp4a_ms) = (tt(0), tt(1), tt(2), tt(3));
        let f0 = Instant::now(); let _ = test_linear_n(0, k, m, &x, t.bytes, iters); let float_ms = f0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        eprintln!("per-dot ms:  float-coop(Tier1) {:.3}  | int 1-thread {:.3}  int __dp4a {:.3}  int warp {:.3}  int warp+dp4a {:.3}", float_ms, plain_ms, dp4a_ms, warp_ms, wdp4a_ms);
        eprintln!("best integer vs float-coop: {:.2}x", float_ms / warp_ms.min(wdp4a_ms).max(1e-6));

        assert_eq!(bit(&gpu_plain, &cpu), m, "GPU plain integer != CPU");
        assert_eq!(bit(&gpu_dp4a, &cpu), m, "GPU __dp4a != CPU");
        assert_eq!(bit(&gpu_warp, &cpu), m, "GPU warp integer != CPU");
        assert_eq!(bit(&gpu_wdp4a, &cpu), m, "GPU warp+dp4a != CPU");
    }

    /// Engagement proof (fast, no CPU recompute): a GPU regime-3 step must differ from a GPU
    /// regime-2 step at the LOGITS level — proving the integer kernels are actually used (the
    /// cert can coincide when the argmax tokens don't flip).
    #[test]
    #[ignore = "needs a Q4_K model + NVIDIA GPU; set VITNI_GGUF_7B"]
    fn cuda_regime3_engages() {
        let path = std::env::var("VITNI_GGUF_7B").expect("set VITNI_GGUF_7B");
        let file = std::fs::File::open(&path).expect("open gguf");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap gguf") };
        let blob: &[u8] = &mmap;
        let gguf = GgufFile::parse(blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        let mut cf2 = CudaForward::new(&cfg, &weights).unwrap();
        let mut cf3 = CudaForward::new_regime(&cfg, &weights, 3).unwrap();
        let l2 = cf2.step(1, 0);
        let l3 = cf3.step(1, 0);
        let diffs = (0..l2.len()).filter(|&i| l2[i].to_bits() != l3[i].to_bits()).count();
        eprintln!("GPU regime-3 vs regime-2 logits: {}/{} differ", diffs, l2.len());
        assert!(diffs > 0, "GPU regime-3 did not engage — logits identical to regime-2");
    }

    /// Increment (b): the GPU regime-3 forward issues a certificate identical to the CPU
    /// regime-3 reference (inference::run_quantized_regime3) — i.e. the integer-activation +
    /// running-sum k-quant linears match end-to-end. Run on a 7B-class Q4_K model.
    #[test]
    #[ignore = "needs a Q4_K model + NVIDIA GPU; set VITNI_GGUF_7B"]
    fn cuda_regime3_matches_cpu_regime3() {
        let path = std::env::var("VITNI_GGUF_7B").expect("set VITNI_GGUF_7B");
        let file = std::fs::File::open(&path).expect("open gguf");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap gguf") };
        let blob: &[u8] = &mmap;
        let hash = *blake3::hash(blob).as_bytes();
        let gguf = GgufFile::parse(blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        eprintln!("device: {}", device_name());
        let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
        let req = Request { model_id: "r3", prompt_tokens: &prompt, n_new_tokens: 12 };

        let cpu = inference::run_quantized_regime3(&cfg, &weights, &hash, &req).unwrap();
        let gpu = run_cuda_regime3(&cfg, &weights, &hash, &req).unwrap();
        // also the shipped regime-2, to confirm regime-3 actually changed the computation
        let cpu2 = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
        eprintln!("regime-2 digest {:02x?}  regime-3 cpu {:02x?}  regime-3 gpu {:02x?}",
            &cpu2.cert.digest[..4], &cpu.cert.digest[..4], &gpu.cert.digest[..4]);
        assert_eq!(cpu.generated_tokens, gpu.generated_tokens, "regime-3: GPU tokens != CPU tokens");
        assert_eq!(cpu.cert.digest, gpu.cert.digest, "regime-3: GPU cert != CPU cert");
        eprintln!("PASS: GPU regime-3 cert == CPU regime-3 cert, bit-for-bit");
    }

    /// The CUDA backend issues a certificate identical to the CPU engine. Run on a
    /// 7B-class model (where the Metal backend's old f32 shortcut diverged):
    ///   VITNI_GGUF_7B=/path/mistral-7b-Q4_K_M.gguf cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore = "needs a model file + an NVIDIA GPU; set VITNI_GGUF_7B"]
    fn cuda_certificate_matches_cpu_7b() {
        let path = std::env::var("VITNI_GGUF_7B").expect("set VITNI_GGUF_7B");
        let blob = std::fs::read(&path).expect("read gguf");
        let hash = *blake3::hash(&blob).as_bytes();
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        eprintln!("device: {}", device_name());
        let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
        let req = Request { model_id: "m7b", prompt_tokens: &prompt, n_new_tokens: 12 };
        let cpu = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
        let gpu = run_cuda(&cfg, &weights, &hash, &req).unwrap();
        assert_eq!(cpu.generated_tokens, gpu.generated_tokens, "GPU generated different tokens");
        assert_eq!(cpu.cert.digest, gpu.cert.digest, "GPU certificate digest differs from CPU");
        eprintln!("PASS: GPU certificate bit-identical to CPU on {}", path);
    }
}
