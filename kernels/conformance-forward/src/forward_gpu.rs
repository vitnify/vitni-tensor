//! Standalone GPU forward for a Llama-2 quantized model (TinyLlama), built from
//! the proven Metal kernels. Host-orchestrated: activations live in host Vecs,
//! each op dispatches to the GPU. Mirrors `forward_quantized::step` exactly so
//! its logits are bit-identical to the CPU engine's — hence the same digest.

use std::ffi::c_void;
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, Buffer, CommandQueue};
use vitni_tensor::model::{config::Config, quant_weights::{QuantizedWeights, QuantTensor}};

fn pipeline(device: &Device, src: &str, func: &str) -> ComputePipelineState {
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let lib = device.new_library_with_source(src, &opts).expect("compile");
    let f = lib.get_function(func, None).expect("fn");
    device.new_compute_pipeline_state_with_function(&f).expect("pipeline")
}

/// An uploaded quantized weight matrix [n, k] (row-major, Q4_K or Q6_K).
struct WBuf {
    buf: Buffer,
    is_q6k: bool,
    n: usize, // output rows
}

pub struct GpuForward {
    device: Device,
    queue: CommandQueue,
    pl_q4k_linear: ComputePipelineState,
    pl_q6k_linear: ComputePipelineState,
    pl_q4k_dequant: ComputePipelineState,
    pl_rms: ComputePipelineState,
    pl_silu: ComputePipelineState,
    pl_rope: ComputePipelineState,
    pl_attn: ComputePipelineState,

    // config
    dim: usize,
    hidden: usize,
    n_layers: usize,
    n_heads: usize,
    kv_dim: usize,
    kv_mul: usize,
    head_size: usize,
    vocab: usize,
    seq_len: usize,
    rms_eps: f32,
    rope_theta: f32,

    // weights
    embed: WBuf,            // [vocab, dim] Q4_K
    wcls: WBuf,             // [vocab, dim]
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

    // host KV cache: [layer][seq*kv_dim]
    key_cache: Vec<Vec<f32>>,
    value_cache: Vec<Vec<f32>>,
}

impl GpuForward {
    pub fn new(cfg: &Config, w: &QuantizedWeights, q4k_src: &str, q6k_int_src: &str, fwd_src: &str) -> GpuForward {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        // n (output rows) comes from the architecture, exactly as the CPU's
        // linear_dispatch is called — NOT from the tensor shape (GGUF stores
        // dims as [n_in, n_out]).
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
        GpuForward {
            pl_q4k_linear: pipeline(&device, q4k_src, "q4k_linear"),
            pl_q6k_linear: pipeline(&device, q6k_int_src, "q6k_integer_linear"),
            pl_q4k_dequant: pipeline(&device, q4k_src, "q4k_dequant"),
            pl_rms: pipeline(&device, fwd_src, "rms_kernel"),
            pl_silu: pipeline(&device, fwd_src, "silu_kernel"),
            pl_rope: pipeline(&device, fwd_src, "rope_apply"),
            pl_attn: pipeline(&device, fwd_src, "attention"),
            dim: cfg.dim,
            hidden: cfg.hidden_dim,
            n_layers,
            n_heads: cfg.n_heads,
            kv_dim: cfg.kv_dim(),
            kv_mul: cfg.kv_mul(),
            head_size: cfg.head_size(),
            vocab: cfg.vocab_size,
            seq_len: cfg.seq_len,
            rms_eps: w.rms_eps,
            rope_theta: w.rope_theta,
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
            key_cache: (0..n_layers).map(|_| vec![0.0f32; cfg.seq_len * cfg.kv_dim()]).collect(),
            value_cache: (0..n_layers).map(|_| vec![0.0f32; cfg.seq_len * cfg.kv_dim()]).collect(),
            device,
            queue,
        }
    }

    fn f32_buf(&self, v: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(v.as_ptr() as *const c_void, (v.len() * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
    }
    fn out_buf(&self, n: usize) -> Buffer {
        self.device.new_buffer((n * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
    }
    fn read(&self, b: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
    }
    fn dispatch(&self, pl: &ComputePipelineState, bufs: &[(&Buffer, u64)], bytes: &[(&[u8], u64)], threads: usize) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        for (b, i) in bufs { enc.set_buffer(*i, Some(b), 0); }
        for (data, i) in bytes { enc.set_bytes(*i, data.len() as u64, data.as_ptr() as *const c_void); }
        let tg = 64u64.min(threads as u64).max(1);
        let groups = (threads as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// out[n] = W[n,k] . input[k], picking the q4k/q6k kernel by dtype.
    fn linear(&self, input: &[f32], wt: &WBuf, k: usize) -> Vec<f32> {
        let bx = self.f32_buf(input);
        let bo = self.out_buf(wt.n);
        let nsuper = k / 256;
        let dims: [u32; 3] = [k as u32, wt.n as u32, nsuper as u32];
        let pl = if wt.is_q6k { &self.pl_q6k_linear } else { &self.pl_q4k_linear };
        self.dispatch(pl, &[(&bx, 0), (&wt.buf, 1), (&bo, 2)], &[(bytemuck(&dims), 3)], wt.n);
        self.read(&bo, wt.n)
    }

    fn rms(&self, x: &[f32], w: &[f32], feat: usize) -> Vec<f32> {
        let bx = self.f32_buf(x);
        let bw = self.f32_buf(w);
        let bo = self.out_buf(feat);
        let dims: [u32; 2] = [feat as u32, 1];
        let eps = self.rms_eps;
        self.dispatch(&self.pl_rms, &[(&bx, 0), (&bw, 1), (&bo, 2)], &[(bytemuck(&dims), 3), (bytemuck(&[eps]), 4)], 1);
        self.read(&bo, feat)
    }

    pub fn embedding_pub(&self, token: usize) -> Vec<f32> { self.embedding(token) }
    fn embedding(&self, token: usize) -> Vec<f32> {
        let nsuper = self.dim / 256;
        let bo = self.out_buf(self.dim);
        let dims: [u32; 1] = [nsuper as u32];
        // set the embedding buffer at the row's byte offset
        let off = (token * nsuper * 144) as u64;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pl_q4k_dequant);
        enc.set_buffer(0, Some(&self.embed.buf), off);
        enc.set_buffer(1, Some(&bo), 0);
        enc.set_bytes(2, 4, dims.as_ptr() as *const c_void);
        let tg = 8u64.min(nsuper as u64).max(1);
        let groups = (nsuper as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        self.read(&bo, self.dim)
    }

    /// rope on a [n_heads_local * head_size] vector using a host cos/sin cache.
    fn rope(&self, x: &[f32], cosc: &[f32], sinc: &[f32], n_heads_local: usize) -> Vec<f32> {
        let bx = self.f32_buf(x);
        let bc = self.f32_buf(cosc);
        let bs = self.f32_buf(sinc);
        let bo = self.out_buf(x.len());
        let dims: [u32; 3] = [1, n_heads_local as u32, self.head_size as u32];
        self.dispatch(&self.pl_rope, &[(&bx, 0), (&bc, 1), (&bs, 2), (&bo, 3)], &[(bytemuck(&dims), 4)], n_heads_local);
        self.read(&bo, x.len())
    }

    fn silu_mul(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        // silu(gate) via kernel, then * up on host (plain mul, exact)
        let bx = self.f32_buf(gate);
        let bo = self.out_buf(gate.len());
        let n = gate.len() as u32;
        self.dispatch(&self.pl_silu, &[(&bx, 0), (&bo, 1)], &[(bytemuck(&[n]), 2)], gate.len());
        let s = self.read(&bo, gate.len());
        s.iter().zip(up).map(|(a, b)| a * b).collect()
    }

    fn attention(&self, q: &[f32], layer: usize, pos: usize) -> Vec<f32> {
        let kc = &self.key_cache[layer][..(pos + 1) * self.kv_dim];
        let vc = &self.value_cache[layer][..(pos + 1) * self.kv_dim];
        let bq = self.f32_buf(q);
        let bk = self.f32_buf(kc);
        let bv = self.f32_buf(vc);
        let bo = self.out_buf(self.n_heads * self.head_size);
        let dims: [u32; 5] = [self.n_heads as u32, self.head_size as u32, self.kv_dim as u32, self.kv_mul as u32, pos as u32];
        self.dispatch(&self.pl_attn, &[(&bq, 0), (&bk, 1), (&bv, 2), (&bo, 3)], &[(bytemuck(&dims), 4)], self.n_heads);
        self.read(&bo, self.n_heads * self.head_size)
    }

    // cos/sin cache for one position, non-neox: inv_freq[j]=1/powf(theta,2j/head_size).
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

    pub fn step(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let dim = self.dim;
        let kv_dim = self.kv_dim;
        let mut x = self.embedding(token as usize);
        let (cosc, sinc) = self.rope_cache(pos);
        for layer in 0..self.n_layers {
            let xb = self.rms(&x, &self.l_rms_att[layer].clone(), dim);
            let q = self.linear(&xb, &self.l_wq[layer], dim);
            let k = self.linear(&xb, &self.l_wk[layer], dim);
            let v = self.linear(&xb, &self.l_wv[layer], dim);
            // cache k,v BEFORE rope
            let koff = pos * kv_dim;
            self.key_cache[layer][koff..koff + kv_dim].copy_from_slice(&k);
            self.value_cache[layer][koff..koff + kv_dim].copy_from_slice(&v);
            // rope q (n_heads) and cached k (kv_dim/head_size heads)
            let q = self.rope(&q, &cosc, &sinc, self.n_heads);
            let k_roped = self.rope(&self.key_cache[layer][koff..koff + kv_dim].to_vec(), &cosc, &sinc, kv_dim / self.head_size);
            self.key_cache[layer][koff..koff + kv_dim].copy_from_slice(&k_roped);
            // attention
            let xb_out = self.attention(&q, layer, pos);
            let xb2 = self.linear(&xb_out, &self.l_wo[layer], dim);
            for i in 0..dim { x[i] += xb2[i]; }
            // ffn
            let xbf = self.rms(&x, &self.l_rms_ffn[layer].clone(), dim);
            let gate = self.linear(&xbf, &self.l_w1[layer], dim);
            let up = self.linear(&xbf, &self.l_w3[layer], dim);
            let inner = self.silu_mul(&gate, &up);
            let down = self.linear(&inner, &self.l_w2[layer], self.hidden);
            for i in 0..dim { x[i] += down[i]; }
        }
        let x = self.rms(&x, &self.rms_final.clone(), dim);
        self.linear(&x, &self.wcls, dim)
    }
}

// tiny helper: reinterpret a &[T] as &[u8] for set_bytes
fn bytemuck<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
