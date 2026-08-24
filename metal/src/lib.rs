//! Metal (Apple GPU) inference backend for `vitni-tensor`.
//!
//! `MetalForward` runs the full quantized Llama-2 forward on the GPU using the
//! bit-exact kernels in `../kernels`. `run_metal` drives it through the engine's
//! `inference::run_quantized_with_forward` seam, so the GPU issues a certificate
//! IDENTICAL to the CPU's — its logits are bit-for-bit identical (verified in
//! `kernels/conformance-forward` on Apple, `kernels/cuda_forward_conformance.cu`
//! on NVIDIA).
//!
//! Weights are uploaded to the GPU once at `new`; each step still round-trips
//! per-op activations through host memory (correctness-first — keeping the
//! residual stream resident on the GPU is the perf follow-up, and does not
//! change the bits).

use std::ffi::c_void;
use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
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

struct WBuf { buf: Buffer, is_q6k: bool, n: usize }

/// A quantized Llama-2 model resident on the Apple GPU.
pub struct MetalForward {
    device: Device,
    queue: CommandQueue,
    pl_q4k_linear: ComputePipelineState,
    pl_q6k_integer: ComputePipelineState,
    pl_q4k_dequant: ComputePipelineState,
    pl_rms: ComputePipelineState,
    pl_silu: ComputePipelineState,
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

    embed: WBuf,
    wcls: WBuf,
    rms_final: Vec<f32>,
    l_rms_att: Vec<Vec<f32>>,
    l_rms_ffn: Vec<Vec<f32>>,
    l_wq: Vec<WBuf>,
    l_wk: Vec<WBuf>,
    l_wv: Vec<WBuf>,
    l_wo: Vec<WBuf>,
    l_w1: Vec<WBuf>,
    l_w2: Vec<WBuf>,
    l_w3: Vec<WBuf>,

    key_cache: Vec<Vec<f32>>,
    value_cache: Vec<Vec<f32>>,
}

const Q4K: &str = include_str!("../../kernels/q4k.metal");
const Q6K_INT: &str = include_str!("../../kernels/q6k_int.metal");
const FWD: &str = include_str!("../../kernels/forward_ops.metal");

impl MetalForward {
    pub fn new(cfg: &Config, w: &QuantizedWeights) -> Result<MetalForward, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let mkw = |t: &QuantTensor, n: usize| -> WBuf {
            let is_q6k = format!("{:?}", t.dtype) == "Q6_K";
            let buf = device.new_buffer_with_data(t.bytes.as_ptr() as *const c_void, t.bytes.len() as u64, MTLResourceOptions::StorageModeShared);
            WBuf { buf, is_q6k, n }
        };
        let n_layers = cfg.n_layers;
        let dim = cfg.dim;
        let kv_dim = cfg.kv_dim();
        let hidden = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        let q_out = cfg.n_heads * cfg.head_size();
        Ok(MetalForward {
            pl_q4k_linear: pipeline(&device, Q4K, "q4k_linear")?,
            pl_q6k_integer: pipeline(&device, Q6K_INT, "q6k_integer_linear")?,
            pl_q4k_dequant: pipeline(&device, Q4K, "q4k_dequant")?,
            pl_rms: pipeline(&device, FWD, "rms_kernel")?,
            pl_silu: pipeline(&device, FWD, "silu_kernel")?,
            pl_rope: pipeline(&device, FWD, "rope_apply")?,
            pl_attn: pipeline(&device, FWD, "attention")?,
            dim, hidden, n_layers,
            n_heads: cfg.n_heads, kv_dim, kv_mul: cfg.kv_mul(), head_size: cfg.head_size(),
            vocab, rms_eps: w.rms_eps, rope_theta: w.rope_theta,
            embed: mkw(&w.token_embedding_table, vocab),
            wcls: match &w.wcls { Some(c) => mkw(c, vocab), None => mkw(&w.token_embedding_table, vocab) },
            rms_final: w.rms_final_weight.to_vec(),
            l_rms_att: w.layers.iter().map(|l| l.rms_att_weight.to_vec()).collect(),
            l_rms_ffn: w.layers.iter().map(|l| l.rms_ffn_weight.to_vec()).collect(),
            l_wq: w.layers.iter().map(|l| mkw(&l.wq, q_out)).collect(),
            l_wk: w.layers.iter().map(|l| mkw(&l.wk, kv_dim)).collect(),
            l_wv: w.layers.iter().map(|l| mkw(&l.wv, kv_dim)).collect(),
            l_wo: w.layers.iter().map(|l| mkw(&l.wo, dim)).collect(),
            l_w1: w.layers.iter().map(|l| mkw(&l.w1, hidden)).collect(),
            l_w2: w.layers.iter().map(|l| mkw(&l.w2, dim)).collect(),
            l_w3: w.layers.iter().map(|l| mkw(&l.w3, hidden)).collect(),
            key_cache: (0..n_layers).map(|_| vec![0.0f32; cfg.seq_len * kv_dim]).collect(),
            value_cache: (0..n_layers).map(|_| vec![0.0f32; cfg.seq_len * kv_dim]).collect(),
            device, queue,
        })
    }

    /// Clear the KV cache to start a fresh sequence.
    pub fn reset(&mut self) {
        for c in self.key_cache.iter_mut().chain(self.value_cache.iter_mut()) {
            for v in c.iter_mut() { *v = 0.0; }
        }
    }

    fn fbuf(&self, v: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(v.as_ptr() as *const c_void, (v.len() * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
    }
    fn obuf(&self, n: usize) -> Buffer { self.device.new_buffer((n * 4).max(4) as u64, MTLResourceOptions::StorageModeShared) }
    fn read(&self, b: &Buffer, n: usize) -> Vec<f32> { unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec() }
    fn go(&self, pl: &ComputePipelineState, bufs: &[(&Buffer, u64)], bytes: &[(&[u8], u64)], threads: usize) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        for (b, i) in bufs { enc.set_buffer(*i, Some(b), 0); }
        for (d, i) in bytes { enc.set_bytes(*i, d.len() as u64, d.as_ptr() as *const c_void); }
        let tg = 64u64.min(threads as u64).max(1);
        let groups = (threads as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn linear(&self, input: &[f32], wt: &WBuf, k: usize) -> Vec<f32> {
        let bx = self.fbuf(input);
        let bo = self.obuf(wt.n);
        let dims = [k as u32, wt.n as u32, (k / 256) as u32];
        let pl = if wt.is_q6k { &self.pl_q6k_integer } else { &self.pl_q4k_linear };
        self.go(pl, &[(&bx, 0), (&wt.buf, 1), (&bo, 2)], &[(bytes(&dims), 3)], wt.n);
        self.read(&bo, wt.n)
    }
    fn rms(&self, x: &[f32], w: &[f32], feat: usize) -> Vec<f32> {
        let bx = self.fbuf(x); let bw = self.fbuf(w); let bo = self.obuf(feat);
        let dims = [feat as u32, 1u32]; let eps = self.rms_eps;
        self.go(&self.pl_rms, &[(&bx, 0), (&bw, 1), (&bo, 2)], &[(bytes(&dims), 3), (bytes(&[eps]), 4)], 1);
        self.read(&bo, feat)
    }
    fn embedding(&self, token: usize) -> Vec<f32> {
        let nsuper = self.dim / 256;
        let bo = self.obuf(self.dim);
        let dims = [nsuper as u32];
        let off = (token * nsuper * 144) as u64;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pl_q4k_dequant);
        enc.set_buffer(0, Some(&self.embed.buf), off);
        enc.set_buffer(1, Some(&bo), 0);
        enc.set_bytes(2, 4, dims.as_ptr() as *const c_void);
        let tg = 8u64.min(nsuper as u64).max(1);
        enc.dispatch_thread_groups(MTLSize::new((nsuper as u64 + tg - 1) / tg, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
        self.read(&bo, self.dim)
    }
    fn rope(&self, x: &[f32], c: &[f32], s: &[f32], n_heads_local: usize) -> Vec<f32> {
        let bx = self.fbuf(x); let bc = self.fbuf(c); let bs = self.fbuf(s); let bo = self.obuf(x.len());
        let dims = [1u32, n_heads_local as u32, self.head_size as u32];
        self.go(&self.pl_rope, &[(&bx, 0), (&bc, 1), (&bs, 2), (&bo, 3)], &[(bytes(&dims), 4)], n_heads_local);
        self.read(&bo, x.len())
    }
    fn silu_mul(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        let bx = self.fbuf(gate); let bo = self.obuf(gate.len()); let n = gate.len() as u32;
        self.go(&self.pl_silu, &[(&bx, 0), (&bo, 1)], &[(bytes(&[n]), 2)], gate.len());
        let s = self.read(&bo, gate.len());
        s.iter().zip(up).map(|(a, b)| a * b).collect()
    }
    fn attention(&self, q: &[f32], layer: usize, pos: usize) -> Vec<f32> {
        let kc = &self.key_cache[layer][..(pos + 1) * self.kv_dim];
        let vc = &self.value_cache[layer][..(pos + 1) * self.kv_dim];
        let bq = self.fbuf(q); let bk = self.fbuf(kc); let bv = self.fbuf(vc);
        let bo = self.obuf(self.n_heads * self.head_size);
        let dims = [self.n_heads as u32, self.head_size as u32, self.kv_dim as u32, self.kv_mul as u32, pos as u32];
        self.go(&self.pl_attn, &[(&bq, 0), (&bk, 1), (&bv, 2), (&bo, 3)], &[(bytes(&dims), 4)], self.n_heads);
        self.read(&bo, self.n_heads * self.head_size)
    }
    fn rope_cache(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_size / 2;
        let mut c = vec![0f32; half]; let mut s = vec![0f32; half];
        for j in 0..half {
            let freq = 1.0f32 / libm::powf(self.rope_theta, (2 * j) as f32 / self.head_size as f32);
            let val = pos as f32 * freq;
            c[j] = libm::cosf(val); s[j] = libm::sinf(val);
        }
        (c, s)
    }

    /// Logits for `token` at sequence position `pos`, advancing the KV cache.
    pub fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let dim = self.dim; let kv_dim = self.kv_dim;
        let mut x = self.embedding(token as usize);
        let (c, s) = self.rope_cache(pos);
        for layer in 0..self.n_layers {
            let xb = self.rms(&x, &self.l_rms_att[layer].clone(), dim);
            let q = self.linear(&xb, &self.l_wq[layer], dim);
            let k = self.linear(&xb, &self.l_wk[layer], dim);
            let v = self.linear(&xb, &self.l_wv[layer], dim);
            let koff = pos * kv_dim;
            self.key_cache[layer][koff..koff + kv_dim].copy_from_slice(&k);
            self.value_cache[layer][koff..koff + kv_dim].copy_from_slice(&v);
            let q = self.rope(&q, &c, &s, self.n_heads);
            let kr = self.rope(&self.key_cache[layer][koff..koff + kv_dim].to_vec(), &c, &s, kv_dim / self.head_size);
            self.key_cache[layer][koff..koff + kv_dim].copy_from_slice(&kr);
            let xo = self.attention(&q, layer, pos);
            let xb2 = self.linear(&xo, &self.l_wo[layer], dim);
            for i in 0..dim { x[i] += xb2[i]; }
            let xf = self.rms(&x, &self.l_rms_ffn[layer].clone(), dim);
            let gate = self.linear(&xf, &self.l_w1[layer], dim);
            let up = self.linear(&xf, &self.l_w3[layer], dim);
            let inner = self.silu_mul(&gate, &up);
            let down = self.linear(&inner, &self.l_w2[layer], self.hidden);
            for i in 0..dim { x[i] += down[i]; }
        }
        let x = self.rms(&x, &self.rms_final.clone(), dim);
        self.linear(&x, &self.wcls, dim)
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
