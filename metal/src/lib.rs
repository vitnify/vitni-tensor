//! Metal (Apple GPU) inference backend for `vitni-tensor`.
//!
//! `MetalForward` runs the full quantized Llama-2 forward on the GPU using the
//! bit-exact kernels in `../kernels`. `run_metal` drives it through the engine's
//! `inference::run_quantized_with_forward` seam, so the GPU issues a certificate
//! IDENTICAL to the CPU's — its logits are bit-for-bit identical (verified in
//! `kernels/conformance-forward` on Apple, `kernels/cuda_forward_conformance.cu`
//! on NVIDIA).
//!
//! The whole model is resident on the GPU: weights + the KV cache + every
//! activation buffer are allocated once, and a decode step encodes all of its
//! ops into a SINGLE command buffer (one CPU<->GPU sync per token, not per op).
//! Only the tiny RoPE cos/sin table is written per step and the logits are read
//! back. This is pure data-movement/batching — the arithmetic (and thus the
//! certificate) is unchanged from the op-at-a-time reference.

use std::ffi::c_void;
use metal::{Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use vitni_tensor::cert::NullSink;
use vitni_tensor::model::config::Config;
use vitni_tensor::model::inference::{self, Outcome, Request, RunError};
use vitni_tensor::model::quant_weights::{QuantTensor, QuantizedWeights};
use vitni_tensor::{Error as TError, Shape, Tensor};

#[derive(Debug)]
pub enum MetalError {
    NoDevice,
    Compile(String),
    Engine(TError),
}
impl core::fmt::Display for MetalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MetalError::NoDevice => write!(f, "no Metal device available"),
            MetalError::Compile(s) => write!(f, "kernel compile: {s}"),
            MetalError::Engine(e) => write!(f, "engine: {e:?}"),
        }
    }
}
impl std::error::Error for MetalError {}
impl From<TError> for MetalError {
    fn from(e: TError) -> Self { MetalError::Engine(e) }
}

fn pipeline(device: &Device, src: &str, func: &str) -> Result<ComputePipelineState, MetalError> {
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let lib = device.new_library_with_source(src, &opts).map_err(MetalError::Compile)?;
    let f = lib.get_function(func, None).map_err(MetalError::Compile)?;
    device.new_compute_pipeline_state_with_function(&f).map_err(MetalError::Compile)
}

/// A quantized weight matrix resident on the GPU. `n` = output rows, `k` = input.
struct WBuf { buf: Buffer, is_q6k: bool, n: usize, k: usize }

const Q4K: &str = include_str!("../../kernels/q4k.metal");
const Q6K_INT: &str = include_str!("../../kernels/q6k_int.metal");
const FWD: &str = include_str!("../../kernels/forward_ops.metal");

/// A quantized Llama-2 model resident on the Apple GPU.
pub struct MetalForward {
    device: Device,
    queue: CommandQueue,
    pl_f32_matvec: ComputePipelineState,
    pl_f32_matvec_acc: ComputePipelineState,
    pl_q6k_quant: ComputePipelineState,
    pl_q6k_dot: ComputePipelineState,
    pl_q6k_dot_acc: ComputePipelineState,
    pl_q4k_dequant: ComputePipelineState,
    pl_rms: ComputePipelineState,
    pl_silu_mul: ComputePipelineState,
    pl_add: ComputePipelineState,
    pl_rope: ComputePipelineState,
    pl_attn: ComputePipelineState,

    dim: usize,
    hidden: usize,
    n_layers: usize,
    n_heads: usize,
    kv_dim: usize,
    kv_mul: usize,
    head_size: usize,
    vocab: usize,
    rms_eps: f32,
    rope_theta: f32,

    // weights
    embed: WBuf,
    wcls: WBuf,
    rms_final: Buffer,
    l_rms_att: Vec<Buffer>,
    l_rms_ffn: Vec<Buffer>,
    l_wq: Vec<WBuf>,
    l_wk: Vec<WBuf>,
    l_wv: Vec<WBuf>,
    l_wo: Vec<WBuf>,
    l_w1: Vec<WBuf>,
    l_w2: Vec<WBuf>,
    l_w3: Vec<WBuf>,

    // KV cache, resident on the GPU: one buffer per layer, [seq_len * kv_dim].
    key_cache: Vec<Buffer>,
    value_cache: Vec<Buffer>,

    // resident activation scratch
    b_x: Buffer,
    b_xb: Buffer,
    b_q: Buffer,
    b_att: Buffer,
    b_tmp_dim: Buffer, // wo output / down output
    b_gate: Buffer,
    b_up: Buffer,
    b_inner: Buffer,
    b_logits: Buffer,
    b_cos: Buffer,
    b_sin: Buffer,
    // scratch for Q6_K quantize-once: x quantized to int8 per super-block, reused
    // by every output row of one linear.
    b_q8_dx: Buffer,
    b_q8_qs: Buffer,
}

fn write_buf(buf: &Buffer, data: &[f32]) {
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len()); }
}

impl MetalForward {
    pub fn new(cfg: &Config, w: &QuantizedWeights) -> Result<MetalForward, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let shared = MTLResourceOptions::StorageModeShared;
        let ubuf = |bytes_: &[u8]| device.new_buffer_with_data(bytes_.as_ptr() as *const c_void, bytes_.len() as u64, shared);
        let fbuf = |v: &[f32]| device.new_buffer_with_data(v.as_ptr() as *const c_void, (v.len() * 4) as u64, shared);
        let zbuf = |n: usize| device.new_buffer((n * 4).max(4) as u64, shared);
        // Q4_K weights are dequantized to f32 ONCE here (bit-identical values to
        // the fused dequant), so the per-token dequant disappears. Q6_K stays
        // quantized (its shipped regime is the integer dot).
        let mkw = |t: &QuantTensor, n: usize, k: usize| {
            let is_q6k = format!("{:?}", t.dtype) == "Q6_K";
            let buf = if is_q6k {
                ubuf(t.bytes)
            } else {
                fbuf(&vitni_tensor::ops::quant::dequantize_q4_k(t.bytes).expect("dequantize Q4_K"))
            };
            WBuf { buf, is_q6k, n, k }
        };

        let dim = cfg.dim;
        let kv_dim = cfg.kv_dim();
        let hidden = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        let head_size = cfg.head_size();
        let q_out = cfg.n_heads * head_size;
        let n_layers = cfg.n_layers;
        let seq = cfg.seq_len;

        Ok(MetalForward {
            pl_f32_matvec: pipeline(&device, Q4K, "f32_matvec")?,
            pl_f32_matvec_acc: pipeline(&device, Q4K, "f32_matvec_acc")?,
            pl_q6k_quant: pipeline(&device, Q6K_INT, "q6k_quantize")?,
            pl_q6k_dot: pipeline(&device, Q6K_INT, "q6k_integer_dot")?,
            pl_q6k_dot_acc: pipeline(&device, Q6K_INT, "q6k_integer_dot_acc")?,
            pl_q4k_dequant: pipeline(&device, Q4K, "q4k_dequant")?,
            pl_rms: pipeline(&device, FWD, "rms_kernel")?,
            pl_silu_mul: pipeline(&device, FWD, "silu_mul")?,
            pl_add: pipeline(&device, FWD, "add_inplace")?,
            pl_rope: pipeline(&device, FWD, "rope_apply")?,
            pl_attn: pipeline(&device, FWD, "attention")?,
            dim, hidden, n_layers,
            n_heads: cfg.n_heads, kv_dim, kv_mul: cfg.kv_mul(), head_size,
            vocab, rms_eps: w.rms_eps, rope_theta: w.rope_theta,
            // embedding stays QUANTIZED — the lookup dequantizes one row per token.
            embed: WBuf { buf: ubuf(w.token_embedding_table.bytes), is_q6k: false, n: vocab, k: dim },
            wcls: match &w.wcls { Some(c) => mkw(c, vocab, dim), None => mkw(&w.token_embedding_table, vocab, dim) },
            rms_final: fbuf(w.rms_final_weight),
            l_rms_att: w.layers.iter().map(|l| fbuf(l.rms_att_weight)).collect(),
            l_rms_ffn: w.layers.iter().map(|l| fbuf(l.rms_ffn_weight)).collect(),
            l_wq: w.layers.iter().map(|l| mkw(&l.wq, q_out, dim)).collect(),
            l_wk: w.layers.iter().map(|l| mkw(&l.wk, kv_dim, dim)).collect(),
            l_wv: w.layers.iter().map(|l| mkw(&l.wv, kv_dim, dim)).collect(),
            l_wo: w.layers.iter().map(|l| mkw(&l.wo, dim, dim)).collect(),
            l_w1: w.layers.iter().map(|l| mkw(&l.w1, hidden, dim)).collect(),
            l_w2: w.layers.iter().map(|l| mkw(&l.w2, dim, hidden)).collect(),
            l_w3: w.layers.iter().map(|l| mkw(&l.w3, hidden, dim)).collect(),
            key_cache: (0..n_layers).map(|_| zbuf(seq * kv_dim)).collect(),
            value_cache: (0..n_layers).map(|_| zbuf(seq * kv_dim)).collect(),
            b_x: zbuf(dim), b_xb: zbuf(dim), b_q: zbuf(q_out), b_att: zbuf(q_out),
            b_tmp_dim: zbuf(dim), b_gate: zbuf(hidden), b_up: zbuf(hidden), b_inner: zbuf(hidden),
            b_logits: zbuf(vocab), b_cos: zbuf(head_size / 2), b_sin: zbuf(head_size / 2),
            b_q8_dx: zbuf(dim.max(hidden) / 256),
            b_q8_qs: device.new_buffer(((dim.max(hidden) / 256) * 256).max(4) as u64, shared),
            device, queue,
        })
    }

    /// Clear the KV cache to start a fresh sequence.
    pub fn reset(&mut self) {
        for c in self.key_cache.iter().chain(self.value_cache.iter()) {
            let n = (c.length() / 4) as usize;
            let z = vec![0.0f32; n];
            write_buf(c, &z);
        }
    }

    /// Encode one dispatch into an existing compute encoder (no encoder switch).
    fn disp(&self, e: &metal::ComputeCommandEncoderRef, pl: &ComputePipelineState, bufs: &[(u64, &Buffer, u64)], scalars: &[(u64, &[u8])], threads: usize) {
        e.set_compute_pipeline_state(pl);
        for (idx, b, off) in bufs { e.set_buffer(*idx, Some(b), *off); }
        for (idx, d) in scalars { e.set_bytes(*idx, d.len() as u64, d.as_ptr() as *const c_void); }
        let tg = 64u64.min(threads as u64).max(1);
        let groups = (threads as u64 + tg - 1) / tg;
        e.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
    }
    /// Order a later dispatch after earlier writes to `bufs` (bit-exact dependency).
    fn barr(&self, e: &metal::ComputeCommandEncoderRef, bufs: &[&Buffer]) {
        let r: Vec<&metal::ResourceRef> = bufs.iter().map(|&b| -> &metal::ResourceRef { b }).collect();
        e.memory_barrier_with_resources(&r);
    }

    /// Encode one quantized linear: out[wt.n] = W . in[wt.k]. Q4_K is a single
    /// dispatch; Q6_K quantizes the input ONCE (shared across all output rows)
    /// then does the integer dot — instead of re-quantizing x per row.
    fn lin(&self, e: &metal::ComputeCommandEncoderRef, in_buf: &Buffer, in_off: u64, wt: &WBuf, out_buf: &Buffer, out_off: u64) {
        let nsuper = wt.k / 256;
        let dims = [wt.k as u32, wt.n as u32, nsuper as u32];
        if wt.is_q6k {
            self.disp(e, &self.pl_q6k_quant,
                &[(0, in_buf, in_off), (1, &self.b_q8_dx, 0), (2, &self.b_q8_qs, 0)],
                &[(3, bytes(&[nsuper as u32]))], nsuper);
            self.barr(e, &[&self.b_q8_dx, &self.b_q8_qs]);
            self.disp(e, &self.pl_q6k_dot,
                &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &wt.buf, 0), (3, out_buf, out_off)],
                &[(4, bytes(&dims))], wt.n);
        } else {
            self.disp(e, &self.pl_f32_matvec, // pre-dequantized f32 weights
                &[(0, in_buf, in_off), (1, &wt.buf, 0), (2, out_buf, out_off)],
                &[(3, bytes(&dims))], wt.n);
        }
    }

    /// Like `lin`, but accumulates: out[i] += (W . in)[i]. Fuses a residual add
    /// into the linear (removes a separate add dispatch). out must already hold
    /// the residual.
    fn lin_acc(&self, e: &metal::ComputeCommandEncoderRef, in_buf: &Buffer, in_off: u64, wt: &WBuf, out_buf: &Buffer, out_off: u64) {
        let nsuper = wt.k / 256;
        let dims = [wt.k as u32, wt.n as u32, nsuper as u32];
        if wt.is_q6k {
            self.disp(e, &self.pl_q6k_quant,
                &[(0, in_buf, in_off), (1, &self.b_q8_dx, 0), (2, &self.b_q8_qs, 0)],
                &[(3, bytes(&[nsuper as u32]))], nsuper);
            self.barr(e, &[&self.b_q8_dx, &self.b_q8_qs]);
            self.disp(e, &self.pl_q6k_dot_acc,
                &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &wt.buf, 0), (3, out_buf, out_off)],
                &[(4, bytes(&dims))], wt.n);
        } else {
            self.disp(e, &self.pl_f32_matvec_acc, // pre-dequantized f32 weights
                &[(0, in_buf, in_off), (1, &wt.buf, 0), (2, out_buf, out_off)],
                &[(3, bytes(&dims))], wt.n);
        }
    }

    fn rope_cache(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_size / 2;
        let mut c = vec![0f32; half];
        let mut s = vec![0f32; half];
        for j in 0..half {
            let freq = 1.0f32 / libm::powf(self.rope_theta, (2 * j) as f32 / self.head_size as f32);
            let val = pos as f32 * freq;
            c[j] = libm::cosf(val);
            s[j] = libm::sinf(val);
        }
        (c, s)
    }

    /// Logits for `token` at sequence position `pos`, advancing the (GPU) KV cache.
    pub fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let (c, s) = self.rope_cache(pos);
        write_buf(&self.b_cos, &c);
        write_buf(&self.b_sin, &s);

        let dim = self.dim;
        let kv_dim = self.kv_dim;
        let koff = (pos * kv_dim * 4) as u64;
        let nsuper_dim = (dim / 256) as u32;
        let head_size = self.head_size;
        let feat_rows = [dim as u32, 1u32];
        let eps = [self.rms_eps];

        let _ = nsuper_dim;
        let cmd = self.queue.new_command_buffer();
        let e = cmd.new_compute_command_encoder();

        // embedding(token) -> b_x
        let nsuper_e = self.dim / 256;
        self.disp(e, &self.pl_q4k_dequant,
            &[(0, &self.embed.buf, (token as usize * nsuper_e * 144) as u64), (1, &self.b_x, 0)],
            &[(2, bytes(&[nsuper_e as u32]))], nsuper_e);
        self.barr(e, &[&self.b_x]);

        for layer in 0..self.n_layers {
            self.disp(e, &self.pl_rms, &[(0, &self.b_x, 0), (1, &self.l_rms_att[layer], 0), (2, &self.b_xb, 0)],
                &[(3, bytes(&feat_rows)), (4, bytes(&eps))], 1);
            self.barr(e, &[&self.b_xb]);
            // q/k/v are independent (all read b_xb) — no barrier between them
            self.lin(e, &self.b_xb, 0, &self.l_wq[layer], &self.b_q, 0);
            self.lin(e, &self.b_xb, 0, &self.l_wk[layer], &self.key_cache[layer], koff);
            self.lin(e, &self.b_xb, 0, &self.l_wv[layer], &self.value_cache[layer], koff);
            self.barr(e, &[&self.b_q, &self.key_cache[layer], &self.value_cache[layer]]);
            self.disp(e, &self.pl_rope, &[(0, &self.b_q, 0), (1, &self.b_cos, 0), (2, &self.b_sin, 0), (3, &self.b_q, 0)],
                &[(4, bytes(&[1u32, self.n_heads as u32, head_size as u32]))], self.n_heads);
            let n_kv = kv_dim / head_size;
            self.disp(e, &self.pl_rope, &[(0, &self.key_cache[layer], koff), (1, &self.b_cos, 0), (2, &self.b_sin, 0), (3, &self.key_cache[layer], koff)],
                &[(4, bytes(&[1u32, n_kv as u32, head_size as u32]))], n_kv);
            self.barr(e, &[&self.b_q, &self.key_cache[layer]]);
            self.disp(e, &self.pl_attn,
                &[(0, &self.b_q, 0), (1, &self.key_cache[layer], 0), (2, &self.value_cache[layer], 0), (3, &self.b_att, 0)],
                &[(4, bytes(&[self.n_heads as u32, head_size as u32, kv_dim as u32, self.kv_mul as u32, pos as u32]))], self.n_heads);
            self.barr(e, &[&self.b_att]);
            self.lin_acc(e, &self.b_att, 0, &self.l_wo[layer], &self.b_x, 0); // wo + residual
            self.barr(e, &[&self.b_x]);
            self.disp(e, &self.pl_rms, &[(0, &self.b_x, 0), (1, &self.l_rms_ffn[layer], 0), (2, &self.b_xb, 0)],
                &[(3, bytes(&feat_rows)), (4, bytes(&eps))], 1);
            self.barr(e, &[&self.b_xb]);
            // gate/up are independent (both read b_xb)
            self.lin(e, &self.b_xb, 0, &self.l_w1[layer], &self.b_gate, 0);
            self.lin(e, &self.b_xb, 0, &self.l_w3[layer], &self.b_up, 0);
            self.barr(e, &[&self.b_gate, &self.b_up]);
            self.disp(e, &self.pl_silu_mul, &[(0, &self.b_gate, 0), (1, &self.b_up, 0), (2, &self.b_inner, 0)],
                &[(3, bytes(&[self.hidden as u32]))], self.hidden);
            self.barr(e, &[&self.b_inner]);
            self.lin_acc(e, &self.b_inner, 0, &self.l_w2[layer], &self.b_x, 0); // w2 (down) + residual
            self.barr(e, &[&self.b_x]);
        }
        self.disp(e, &self.pl_rms, &[(0, &self.b_x, 0), (1, &self.rms_final, 0), (2, &self.b_xb, 0)],
            &[(3, bytes(&feat_rows)), (4, bytes(&eps))], 1);
        self.barr(e, &[&self.b_xb]);
        self.lin(e, &self.b_xb, 0, &self.wcls, &self.b_logits, 0);

        e.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { std::slice::from_raw_parts(self.b_logits.contents() as *const f32, self.vocab) }.to_vec()
    }
}

fn bytes<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Run a full generation on the Apple GPU and issue a certificate. The result
/// (tokens + certificate) is identical to `inference::run_quantized` on CPU,
/// because the GPU forward is bit-identical.
pub fn run_metal(
    cfg: &Config,
    weights: &QuantizedWeights,
    weights_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome, MetalError> {
    let mut mf = MetalForward::new(cfg, weights)?;
    let vocab = cfg.vocab_size;
    let (outcome, _id) = inference::run_quantized_with_forward(
        cfg, weights_hash, req, &mut NullSink, 0,
        |cur, pos| Tensor::from_f32(mf.step(cur, pos), Shape::new(&[vocab])?),
    )
    .map_err(|e| match e {
        RunError::Inference(err) => MetalError::Engine(err),
        RunError::Sink(_) => unreachable!("NullSink is infallible"),
    })?;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vitni_tensor::model::gguf::GgufFile;

    /// Rough throughput: GPU (resident, one command buffer/token) vs CPU.
    #[test]
    #[ignore = "benchmark; needs a model file"]
    fn bench_tokens_per_sec() {
        use std::time::Instant;
        use vitni_tensor::model::{forward::RunState, forward_quantized};
        let path = std::env::var("VITNI_GGUF")
            .unwrap_or_else(|_| "/Users/nickp/Downloads/vitnify_test/tinyllama-Q4_K_M.gguf".into());
        let blob = std::fs::read(&path).expect("read gguf");
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        let tok = 1u32;
        let n = 32usize;

        let mut mf = MetalForward::new(&cfg, &weights).unwrap();
        for pos in 0..4 { mf.step(tok, pos); } // warmup
        let t = Instant::now();
        for pos in 4..4 + n { let _ = mf.step(tok, pos); }
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let mut st = RunState::new(&cfg);
        for pos in 0..4 { forward_quantized::step(&cfg, &weights, &mut st, tok, pos).unwrap(); }
        let t = Instant::now();
        for pos in 4..4 + n { let _ = forward_quantized::step(&cfg, &weights, &mut st, tok, pos).unwrap(); }
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;

        eprintln!("GPU (Metal, resident): {:.1} ms/tok  ({:.1} tok/s)", gpu_ms, 1000.0 / gpu_ms);
        eprintln!("CPU (single-thread):   {:.1} ms/tok  ({:.1} tok/s)", cpu_ms, 1000.0 / cpu_ms);
    }

    /// The Apple GPU backend issues a certificate identical to the CPU engine.
    #[test]
    #[ignore = "needs a model file; run with VITNI_GGUF set (default: the local TinyLlama)"]
    fn metal_certificate_matches_cpu() {
        let path = std::env::var("VITNI_GGUF")
            .unwrap_or_else(|_| "/Users/nickp/Downloads/vitnify_test/tinyllama-Q4_K_M.gguf".into());
        let blob = std::fs::read(&path).expect("read gguf");
        let hash = *blake3::hash(&blob).as_bytes();
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
        let req = Request { model_id: "tinyllama-1.1b-chat-Q4_K_M", prompt_tokens: &prompt, n_new_tokens: 16 };

        let cpu = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
        let gpu = run_metal(&cfg, &weights, &hash, &req).unwrap();

        assert_eq!(cpu.generated_tokens, gpu.generated_tokens, "GPU generated different tokens");
        assert_eq!(cpu.cert.digest, gpu.cert.digest, "GPU certificate digest differs from CPU");
    }
}
