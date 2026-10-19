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
/// `kind`: 0 = Q4_K, 1 = Q6_K, 2 = Q8_0, 3 = Q5_0 (the dtypes these models' linears use).
struct WBuf { buf: Buffer, kind: u8, n: usize, k: usize }

/// Map a weight tensor's GGUF dtype to a WBuf `kind`. Unknown dtypes fall back to 0
/// (Q4_K); the models in scope (TinyLlama, Mistral, Qwen2.5) only use 0–3.
fn dtype_kind(dtype_dbg: &str) -> u8 {
    match dtype_dbg { "Q6_K" => 1, "Q8_0" => 2, "Q5_0" => 3, _ => 0 }
}

const Q4K: &str = include_str!("../../kernels/q4k.metal");
const Q6K_INT: &str = include_str!("../../kernels/q6k_int.metal");
const FWD: &str = include_str!("../../kernels/forward_ops.metal");
const REGIME3: &str = include_str!("../../kernels/regime3.metal");
const DTYPES: &str = include_str!("../../kernels/dtypes.metal");

/// A quantized Llama-2 model resident on the Apple GPU.
pub struct MetalForward {
    device: Device,
    queue: CommandQueue,
    pl_q4k_linear: ComputePipelineState,
    pl_q4k_linear_acc: ComputePipelineState,
    pl_q6k_quant: ComputePipelineState,
    pl_q6k_dot: ComputePipelineState,
    pl_q6k_dot_acc: ComputePipelineState,
    pl_q4k_dequant: ComputePipelineState,
    pl_rms: ComputePipelineState,
    pl_silu_mul: ComputePipelineState,
    pl_add: ComputePipelineState,
    pl_rope: ComputePipelineState,
    pl_rope_neox: ComputePipelineState,   // Qwen2 split-half RoPE
    pl_attn: ComputePipelineState,
    rope_neox: bool,                      // true = NeoX (Qwen2), false = interleaved (Llama)
    // Q8_0 / Q5_0 linears (dequant row + canonical dot) — regime-independent
    pl_q8_0: ComputePipelineState,
    pl_q8_0_acc: ComputePipelineState,
    pl_q5_0: ComputePipelineState,
    pl_q5_0_acc: ComputePipelineState,
    // regime-3 (reserved): integer-activation k-quant linears + running-sum combine
    pl_q8_quant: ComputePipelineState,
    pl_q4k_rs: ComputePipelineState,
    pl_q4k_rs_acc: ComputePipelineState,
    pl_q6k_rs: ComputePipelineState,
    pl_q6k_rs_acc: ComputePipelineState,
    regime: u8,   // 2 = shipped float Q4_K + fixed_tree; 3 = integer + running-sum

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
    // Embedding lookup is dequantized on the HOST (per token, any dtype) — matches the
    // CPU `embedding_row` bit-for-bit, so it works for Q4_K/Q6_K/Q8_0/Q5_0/Q4_0/F32 embeds.
    embed_host: Vec<u8>,
    embed_dtype: String,
    wcls: WBuf,
    rms_final: Buffer,
    l_rms_att: Vec<Buffer>,
    l_rms_ffn: Vec<Buffer>,
    l_wq: Vec<WBuf>,
    l_wk: Vec<WBuf>,
    l_wv: Vec<WBuf>,
    // optional QKV projection biases (Qwen2) as f32 device buffers; None for Llama
    l_bq: Vec<Option<Buffer>>,
    l_bk: Vec<Option<Buffer>>,
    l_bv: Vec<Option<Buffer>>,
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
    b_q8_bsums: Buffer,   // regime-3: per-32-group int sums of the int8 activation
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
        // Both Q4_K and Q6_K stay QUANTIZED on the GPU. The fused q4k_linear /
        // q6k_dot kernels dequantize in-kernel with the canonical reduction, so
        // they are bit-identical to the CPU hot path (proven in
        // kernels/conformance-forward). The old "dequantize Q4_K to f32 once +
        // f32_matvec" shortcut diverged from the fused reduction at 7B matrix
        // sizes (TinyLlama agreed by luck; Mistral-7B did not). Keeping weights
        // quantized also avoids the f32 memory blow-up.
        let mkw = |t: &QuantTensor, n: usize, k: usize| {
            WBuf { buf: ubuf(t.bytes), kind: dtype_kind(&format!("{:?}", t.dtype)), n, k }
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
            pl_q4k_linear: pipeline(&device, Q4K, "q4k_linear")?,
            pl_q4k_linear_acc: pipeline(&device, Q4K, "q4k_linear_acc")?,
            pl_q6k_quant: pipeline(&device, Q6K_INT, "q6k_quantize")?,
            pl_q6k_dot: pipeline(&device, Q6K_INT, "q6k_integer_dot")?,
            pl_q6k_dot_acc: pipeline(&device, Q6K_INT, "q6k_integer_dot_acc")?,
            pl_q4k_dequant: pipeline(&device, Q4K, "q4k_dequant")?,
            pl_rms: pipeline(&device, FWD, "rms_kernel")?,
            pl_silu_mul: pipeline(&device, FWD, "silu_mul")?,
            pl_add: pipeline(&device, FWD, "add_inplace")?,
            pl_rope: pipeline(&device, FWD, "rope_apply")?,
            pl_rope_neox: pipeline(&device, FWD, "rope_apply_neox")?,
            rope_neox: w.rope_neox,
            pl_attn: pipeline(&device, FWD, "attention")?,
            pl_q8_0: pipeline(&device, DTYPES, "q8_0_linear")?,
            pl_q8_0_acc: pipeline(&device, DTYPES, "q8_0_linear_acc")?,
            pl_q5_0: pipeline(&device, DTYPES, "q5_0_linear")?,
            pl_q5_0_acc: pipeline(&device, DTYPES, "q5_0_linear_acc")?,
            pl_q8_quant: pipeline(&device, REGIME3, "q8_quant")?,
            pl_q4k_rs: pipeline(&device, REGIME3, "q4k_integer_rs")?,
            pl_q4k_rs_acc: pipeline(&device, REGIME3, "q4k_integer_rs_acc")?,
            pl_q6k_rs: pipeline(&device, REGIME3, "q6k_integer_rs")?,
            pl_q6k_rs_acc: pipeline(&device, REGIME3, "q6k_integer_rs_acc")?,
            regime: 2,
            dim, hidden, n_layers,
            n_heads: cfg.n_heads, kv_dim, kv_mul: cfg.kv_mul(), head_size,
            vocab, rms_eps: w.rms_eps, rope_theta: w.rope_theta,
            // embedding kept on the HOST — the lookup dequantizes one row per token for any dtype.
            embed_host: w.token_embedding_table.bytes.to_vec(),
            embed_dtype: format!("{:?}", w.token_embedding_table.dtype),
            wcls: match &w.wcls { Some(c) => mkw(c, vocab, dim), None => mkw(&w.token_embedding_table, vocab, dim) },
            rms_final: fbuf(w.rms_final_weight),
            l_rms_att: w.layers.iter().map(|l| fbuf(l.rms_att_weight)).collect(),
            l_rms_ffn: w.layers.iter().map(|l| fbuf(l.rms_ffn_weight)).collect(),
            l_wq: w.layers.iter().map(|l| mkw(&l.wq, q_out, dim)).collect(),
            l_wk: w.layers.iter().map(|l| mkw(&l.wk, kv_dim, dim)).collect(),
            l_wv: w.layers.iter().map(|l| mkw(&l.wv, kv_dim, dim)).collect(),
            l_bq: w.layers.iter().map(|l| l.bq.map(|b| fbuf(b))).collect(),
            l_bk: w.layers.iter().map(|l| l.bk.map(|b| fbuf(b))).collect(),
            l_bv: w.layers.iter().map(|l| l.bv.map(|b| fbuf(b))).collect(),
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
            b_q8_bsums: device.new_buffer((((dim.max(hidden) / 256) * 8) * 4).max(4) as u64, shared),
            device, queue,
        })
    }

    /// Select the numeric regime: 2 = shipped (float Q4_K + fixed_tree), 3 = reserved
    /// integer-activation k-quant linears with running-sum combine (bit-identical to the
    /// CPU `run_quantized_regime3`). Only changes how Q4_K/Q6_K linears are computed.
    pub fn set_regime(&mut self, regime: u8) { self.regime = regime; }

    /// Debug: run ONLY the layer's w2 (ffn_down) linear on `x` (hidden-dim) under the
    /// current regime, returning the dim-length output. Isolates one linear for testing.
    pub fn debug_w2(&mut self, x: &[f32], layer: usize) -> Vec<f32> {
        write_buf(&self.b_inner, x);
        let z = vec![0.0f32; self.dim];
        write_buf(&self.b_tmp_dim, &z);
        let cmd = self.queue.new_command_buffer();
        let e = cmd.new_compute_command_encoder();
        self.lin(e, &self.b_inner, 0, &self.l_w2[layer], &self.b_tmp_dim, 0);
        e.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { std::slice::from_raw_parts(self.b_tmp_dim.contents() as *const f32, self.dim) }.to_vec()
    }

    /// Debug: w2 with accumulation — out = residual + (w2 . x), the form the forward uses.
    pub fn debug_w2_acc(&mut self, x: &[f32], residual: &[f32], layer: usize) -> Vec<f32> {
        write_buf(&self.b_inner, x);
        write_buf(&self.b_tmp_dim, residual);
        let cmd = self.queue.new_command_buffer();
        let e = cmd.new_compute_command_encoder();
        self.lin_acc(e, &self.b_inner, 0, &self.l_w2[layer], &self.b_tmp_dim, 0);
        e.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { std::slice::from_raw_parts(self.b_tmp_dim.contents() as *const f32, self.dim) }.to_vec()
    }

    /// Dequantize one embedding row on the host (any quant type), matching the CPU
    /// `embedding_row` bit-for-bit — so Q8_0/Q5_0/Q6_K/Q4_K/Q4_0/F32 embeddings all work.
    fn embed_row(&self, token: usize) -> Vec<f32> {
        use vitni_tensor::ops::quant::{dequantize_q4_0, dequantize_q4_k, dequantize_q5_0, dequantize_q6_k, dequantize_q8_0};
        let dim = self.dim;
        let b = &self.embed_host;
        let slice = |rb: usize| -> &[u8] { &b[token * rb..(token + 1) * rb] };
        match self.embed_dtype.as_str() {
            "F32" => { let off = token * dim * 4; (0..dim).map(|i| f32::from_le_bytes([b[off + i * 4], b[off + i * 4 + 1], b[off + i * 4 + 2], b[off + i * 4 + 3]])).collect() }
            "Q4_0" => dequantize_q4_0(slice((dim / 32) * 18)).expect("q4_0 embed row"),
            "Q8_0" => dequantize_q8_0(slice((dim / 32) * 34)).expect("q8_0 embed row"),
            "Q5_0" => dequantize_q5_0(slice((dim / 32) * 22)).expect("q5_0 embed row"),
            "Q4_K" => dequantize_q4_k(slice((dim / 256) * 144)).expect("q4_k embed row"),
            "Q6_K" => dequantize_q6_k(slice((dim / 256) * 210)).expect("q6_k embed row"),
            other => panic!("metal: unsupported embedding dtype {other}"),
        }
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
        // Q8_0 / Q5_0: dequant each row + canonical dot (regime-independent).
        if wt.kind == 2 {
            self.disp(e, &self.pl_q8_0, &[(0, in_buf, in_off), (1, &wt.buf, 0), (2, out_buf, out_off)], &[(3, bytes(&dims))], wt.n);
            return;
        }
        if wt.kind == 3 {
            self.disp(e, &self.pl_q5_0, &[(0, in_buf, in_off), (1, &wt.buf, 0), (2, out_buf, out_off)], &[(3, bytes(&dims))], wt.n);
            return;
        }
        if self.regime == 3 {
            self.disp(e, &self.pl_q8_quant,
                &[(0, in_buf, in_off), (1, &self.b_q8_dx, 0), (2, &self.b_q8_qs, 0), (3, &self.b_q8_bsums, 0)],
                &[(4, bytes(&[nsuper as u32]))], nsuper);
            self.barr(e, &[&self.b_q8_dx, &self.b_q8_qs, &self.b_q8_bsums]);
            if wt.kind == 1 {
                self.disp(e, &self.pl_q6k_rs,
                    &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &wt.buf, 0), (3, out_buf, out_off)],
                    &[(4, bytes(&dims))], wt.n);
            } else {
                self.disp(e, &self.pl_q4k_rs,
                    &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &self.b_q8_bsums, 0), (3, &wt.buf, 0), (4, out_buf, out_off)],
                    &[(5, bytes(&dims))], wt.n);
            }
            return;
        }
        if wt.kind == 1 {
            self.disp(e, &self.pl_q6k_quant,
                &[(0, in_buf, in_off), (1, &self.b_q8_dx, 0), (2, &self.b_q8_qs, 0)],
                &[(3, bytes(&[nsuper as u32]))], nsuper);
            self.barr(e, &[&self.b_q8_dx, &self.b_q8_qs]);
            self.disp(e, &self.pl_q6k_dot,
                &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &wt.buf, 0), (3, out_buf, out_off)],
                &[(4, bytes(&dims))], wt.n);
        } else {
            self.disp(e, &self.pl_q4k_linear, // fused dequant + canonical dot on quantized Q4_K
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
        if wt.kind == 2 {
            self.disp(e, &self.pl_q8_0_acc, &[(0, in_buf, in_off), (1, &wt.buf, 0), (2, out_buf, out_off)], &[(3, bytes(&dims))], wt.n);
            return;
        }
        if wt.kind == 3 {
            self.disp(e, &self.pl_q5_0_acc, &[(0, in_buf, in_off), (1, &wt.buf, 0), (2, out_buf, out_off)], &[(3, bytes(&dims))], wt.n);
            return;
        }
        if self.regime == 3 {
            self.disp(e, &self.pl_q8_quant,
                &[(0, in_buf, in_off), (1, &self.b_q8_dx, 0), (2, &self.b_q8_qs, 0), (3, &self.b_q8_bsums, 0)],
                &[(4, bytes(&[nsuper as u32]))], nsuper);
            self.barr(e, &[&self.b_q8_dx, &self.b_q8_qs, &self.b_q8_bsums]);
            if wt.kind == 1 {
                self.disp(e, &self.pl_q6k_rs_acc,
                    &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &wt.buf, 0), (3, out_buf, out_off)],
                    &[(4, bytes(&dims))], wt.n);
            } else {
                self.disp(e, &self.pl_q4k_rs_acc,
                    &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &self.b_q8_bsums, 0), (3, &wt.buf, 0), (4, out_buf, out_off)],
                    &[(5, bytes(&dims))], wt.n);
            }
            return;
        }
        if wt.kind == 1 {
            self.disp(e, &self.pl_q6k_quant,
                &[(0, in_buf, in_off), (1, &self.b_q8_dx, 0), (2, &self.b_q8_qs, 0)],
                &[(3, bytes(&[nsuper as u32]))], nsuper);
            self.barr(e, &[&self.b_q8_dx, &self.b_q8_qs]);
            self.disp(e, &self.pl_q6k_dot_acc,
                &[(0, &self.b_q8_dx, 0), (1, &self.b_q8_qs, 0), (2, &wt.buf, 0), (3, out_buf, out_off)],
                &[(4, bytes(&dims))], wt.n);
        } else {
            self.disp(e, &self.pl_q4k_linear_acc, // fused dequant + canonical dot on quantized Q4_K
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
        // embedding(token) -> b_x, dequantized on the host (any dtype), bit-for-bit == CPU.
        let erow = self.embed_row(token as usize);
        write_buf(&self.b_x, &erow);

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

        for layer in 0..self.n_layers {
            self.disp(e, &self.pl_rms, &[(0, &self.b_x, 0), (1, &self.l_rms_att[layer], 0), (2, &self.b_xb, 0)],
                &[(3, bytes(&feat_rows)), (4, bytes(&eps))], 1);
            self.barr(e, &[&self.b_xb]);
            // q/k/v are independent (all read b_xb) — no barrier between them
            self.lin(e, &self.b_xb, 0, &self.l_wq[layer], &self.b_q, 0);
            self.lin(e, &self.b_xb, 0, &self.l_wk[layer], &self.key_cache[layer], koff);
            self.lin(e, &self.b_xb, 0, &self.l_wv[layer], &self.value_cache[layer], koff);
            self.barr(e, &[&self.b_q, &self.key_cache[layer], &self.value_cache[layer]]);
            // optional QKV bias (Qwen2): q/k/v += bias, after projection and BEFORE rope.
            if self.l_bq[layer].is_some() {
                let qo = self.n_heads * head_size;
                self.disp(e, &self.pl_add, &[(0, &self.b_q, 0), (1, self.l_bq[layer].as_ref().unwrap(), 0)], &[(2, bytes(&[qo as u32]))], qo);
                self.disp(e, &self.pl_add, &[(0, &self.key_cache[layer], koff), (1, self.l_bk[layer].as_ref().unwrap(), 0)], &[(2, bytes(&[kv_dim as u32]))], kv_dim);
                self.disp(e, &self.pl_add, &[(0, &self.value_cache[layer], koff), (1, self.l_bv[layer].as_ref().unwrap(), 0)], &[(2, bytes(&[kv_dim as u32]))], kv_dim);
                self.barr(e, &[&self.b_q, &self.key_cache[layer], &self.value_cache[layer]]);
            }
            let rope = if self.rope_neox { &self.pl_rope_neox } else { &self.pl_rope };
            self.disp(e, rope, &[(0, &self.b_q, 0), (1, &self.b_cos, 0), (2, &self.b_sin, 0), (3, &self.b_q, 0)],
                &[(4, bytes(&[1u32, self.n_heads as u32, head_size as u32]))], self.n_heads);
            let n_kv = kv_dim / head_size;
            self.disp(e, rope, &[(0, &self.key_cache[layer], koff), (1, &self.b_cos, 0), (2, &self.b_sin, 0), (3, &self.key_cache[layer], koff)],
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

/// Run a full generation on the Apple GPU in **regime-3** (reserved: integer-activation
/// k-quant linears + running-sum combine). Bit-identical to `inference::run_quantized_regime3`
/// on CPU — the Apple-GPU half of the cross-vendor regime-3 proof (NVIDIA is the CUDA backend).
pub fn run_metal_regime3(
    cfg: &Config,
    weights: &QuantizedWeights,
    weights_hash: &[u8; 32],
    req: &Request,
) -> Result<Outcome, MetalError> {
    let mut mf = MetalForward::new(cfg, weights)?;
    mf.set_regime(3);
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

    /// Isolate the w2 (Q6_K) regime-3 linear: Metal `q8_quant + q6k_integer_rs` vs the CPU
    /// `linear_q6_k_integer_rs`, on a controlled hidden-dim input. Pinpoints a per-linear diff.
    #[test]
    #[ignore = "needs a Q6_K-w2 model; set VITNI_GGUF (default: local Qwen2.5-0.5B)"]
    fn metal_w2_q6k_rs_isolation() {
        use vitni_tensor::ops::quant::linear_q6_k_integer_rs;
        let path = std::env::var("VITNI_GGUF")
            .unwrap_or_else(|_| "/Users/nickp/Downloads/vitnify_test/qwen2.5-0.5b-Q4_K_M.gguf".into());
        let blob = std::fs::read(&path).expect("read gguf");
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        use vitni_tensor::ops::quant::linear_q4_k_integer_rs;
        let k = cfg.hidden_dim; let m = cfg.dim;
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.6180339887).fract() - 0.5) * 2.0).collect();
        let mut mf = MetalForward::new(&cfg, &weights).unwrap();
        mf.set_regime(3);
        let mut bad = 0usize;
        for layer in 0..weights.layers.len() {
            let w2 = &weights.layers[layer].w2;
            let dt = format!("{:?}", w2.dtype);
            let mut cpu = vec![0f32; m];
            if dt == "Q6_K" { linear_q6_k_integer_rs(&x, w2.bytes, &mut cpu, 1, k, m).unwrap(); }
            else { linear_q4_k_integer_rs(&x, w2.bytes, &mut cpu, 1, k, m).unwrap(); }
            let gpu = mf.debug_w2(&x, layer);
            let diffs = (0..m).filter(|&i| cpu[i].to_bits() != gpu[i].to_bits()).count();
            if diffs > 0 { eprintln!("layer {} w2 {}: {}/{} diffs", layer, dt, diffs, m); bad += 1; }
        }
        assert_eq!(bad, 0, "some w2 regime-3 linears diverge from CPU (Q4_K/Q6_K integer-rs)");
    }

    /// Cross-vendor regime-3: the Apple Metal GPU, in the reserved integer-activation regime,
    /// reproduces the CPU `run_quantized_regime3` certificate bit-for-bit — the same regime-3
    /// the NVIDIA CUDA backend reproduces. Also asserts regime-3 actually engages (its cert
    /// differs from regime-2's when the integer activation flips the trajectory).
    #[test]
    #[ignore = "needs a model file; run with VITNI_GGUF set (default: the local TinyLlama)"]
    fn metal_regime3_matches_cpu() {
        let path = std::env::var("VITNI_GGUF")
            .unwrap_or_else(|_| "/Users/nickp/Downloads/vitnify_test/tinyllama-Q4_K_M.gguf".into());
        let blob = std::fs::read(&path).expect("read gguf");
        let hash = *blake3::hash(&blob).as_bytes();
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
        let req = Request { model_id: "tinyllama-1.1b-chat-Q4_K_M", prompt_tokens: &prompt, n_new_tokens: 16 };

        let cpu3 = inference::run_quantized_regime3(&cfg, &weights, &hash, &req).unwrap();
        let gpu3 = run_metal_regime3(&cfg, &weights, &hash, &req).unwrap();
        let cpu2 = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
        eprintln!("metal r3 {:02x?}  cpu r3 {:02x?}  (cpu r2 {:02x?})",
            &gpu3.cert.digest[..4], &cpu3.cert.digest[..4], &cpu2.cert.digest[..4]);

        // ENGAGEMENT (logits level): the Metal regime-3 kernels must actually change the
        // computation vs regime-2 — otherwise a matching cert could be regime-2 in disguise,
        // since the generated tokens can coincide. Compare step-0 logits on the same GPU.
        let mut mf = MetalForward::new(&cfg, &weights).unwrap();
        let l2 = mf.step(prompt[0], 0);
        mf.reset();
        mf.set_regime(3);
        let l3 = mf.step(prompt[0], 0);
        let diffs = l2.iter().zip(l3.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("metal regime-3 engagement: {}/{} step-0 logits differ from regime-2", diffs, l2.len());
        assert!(diffs > 0, "Metal regime-3 logits == regime-2 — integer kernels NOT engaged");

        assert_eq!(cpu3.generated_tokens, gpu3.generated_tokens, "Metal regime-3 tokens differ from CPU regime-3");
        assert_eq!(cpu3.cert.digest, gpu3.cert.digest, "Metal regime-3 cert differs from CPU regime-3");
    }

    /// Regression guard for a bug that only showed at LARGE matrix sizes. MetalForward
    /// used to pre-dequantize Q4_K to f32 and run f32_matvec; that diverged from the
    /// fused canonical reduction at 7B dimensions (Mistral: dim 4096, FFN 14336), while
    /// TinyLlama's smaller dims agreed by luck. Testing only TinyLlama missed it. Run
    /// this on a 7B-class model whenever one is available.
    #[test]
    #[ignore = "needs a 7B-class model; set VITNI_GGUF_7B (e.g. Mistral-7B Q4_K_M)"]
    fn metal_certificate_matches_cpu_7b() {
        let path = match std::env::var("VITNI_GGUF_7B") {
            Ok(p) => p,
            Err(_) => { eprintln!("VITNI_GGUF_7B not set; skipping 7B bit-identity check"); return; }
        };
        let blob = std::fs::read(&path).expect("read gguf");
        let hash = *blake3::hash(&blob).as_bytes();
        let gguf = GgufFile::parse(&blob).unwrap();
        let cfg = Config::from_gguf(&gguf).unwrap();
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
        let prompt: Vec<u32> = vec![1, 9038, 2501, 263, 931, 29892];
        let req = Request { model_id: "m7b", prompt_tokens: &prompt, n_new_tokens: 12 };
        let cpu = inference::run_quantized(&cfg, &weights, &hash, &req).unwrap();
        let gpu = run_metal(&cfg, &weights, &hash, &req).unwrap();
        assert_eq!(cpu.generated_tokens, gpu.generated_tokens, "7B: GPU generated different tokens");
        assert_eq!(cpu.cert.digest, gpu.cert.digest, "7B: GPU certificate digest differs from CPU");
    }
}
