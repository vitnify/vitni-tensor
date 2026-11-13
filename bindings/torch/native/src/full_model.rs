//! Resident full-engine bridge. Numerical work stays in the existing forward drivers.
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::CString,
    panic::catch_unwind,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};
use vitni_tensor::{
    model::{
        config::Config,
        forward::RunState,
        forward_quantized,
        gguf::{GgufFile, GgufTensorType},
        inference,
        quant_weights::QuantizedWeights,
    },
    Shape, Storage, Tensor,
};

type Result<T> = std::result::Result<T, String>;
enum Backend {
    // Keep original weight bytes resident; rebuild borrowing views per request
    // without unsafe self-references, file I/O, rehashing or copying matrices.
    Cpu {
        blob: Vec<u8>,
        state: RunState,
    },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(Box<vitni_tensor_metal::MetalForward>),
}

pub struct Session {
    pub cfg: Config,
    pub weights_hash: [u8; 32],
    backend: Backend,
    optimized: bool,
    forward_calls: u64,
    generate_calls: u64,
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl Session {
    pub fn open(path: &str, backend: u32, optimized: bool) -> Result<Self> {
        if !super::environment_ok() {
            return Err("unsupported CPU floating-point environment".into());
        }
        if backend > 1 {
            return Err("backend must be CPU (0) or Metal (1)".into());
        }
        let blob = std::fs::read(path).map_err(|e| format!("read GGUF: {e}"))?;
        let weights_hash = *blake3::hash(&blob).as_bytes();
        let gguf = GgufFile::parse(&blob).map_err(|e| format!("parse GGUF: {e:?}"))?;
        let cfg = Config::from_gguf(&gguf).map_err(|e| format!("model config: {e:?}"))?;
        let weights = QuantizedWeights::from_gguf(&gguf, &cfg)
            .map_err(|e| format!("model weights: {e:?}"))?;
        let backend = if backend == 0 {
            Backend::Cpu {
                blob,
                state: RunState::new(&cfg),
            }
        } else {
            // These are the same bounds as the released standalone Metal CLI.
            if cfg.dim > 8192 || cfg.head_size() > 8192 || cfg.hidden_dim > 64 * 8192 {
                return Err("unsupported Metal model dimensions".into());
            }
            let supported =
                |t: &vitni_tensor::model::quant_weights::QuantTensor, k: usize| match t.dtype {
                    GgufTensorType::Q4_K | GgufTensorType::Q6_K => k % 256 == 0,
                    GgufTensorType::Q5_0 | GgufTensorType::Q8_0 => k % 32 == 0,
                    _ => false,
                };
            for layer in &weights.layers {
                if [
                    &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w1, &layer.w3,
                ]
                .iter()
                .any(|t| !supported(t, cfg.dim))
                    || !supported(&layer.w2, cfg.hidden_dim)
                {
                    return Err("unsupported Metal projection dtype/block width".into());
                }
            }
            if !supported(
                weights
                    .wcls
                    .as_ref()
                    .unwrap_or(&weights.token_embedding_table),
                cfg.dim,
            ) {
                return Err("unsupported Metal classifier dtype/block width".into());
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let options = if optimized {
                    vitni_tensor_metal::MetalOptions::optimized()
                } else {
                    Default::default()
                };
                Backend::Metal(Box::new(
                    metal_pool(|| {
                        vitni_tensor_metal::MetalForward::new_with_options(&cfg, &weights, &options)
                    })
                    .map_err(|e| e.to_string())?,
                ))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                return Err("Metal support is not compiled into this binding".into());
            }
        };
        Ok(Self {
            cfg,
            weights_hash,
            backend,
            optimized,
            forward_calls: 0,
            generate_calls: 0,
        })
    }

    pub fn validate(&self, tokens: &[i64], extra: usize) -> Result<Vec<u32>> {
        if tokens.is_empty()
            || tokens
                .len()
                .checked_add(extra)
                .is_none_or(|n| n > self.cfg.seq_len.min(512))
        {
            return Err(
                "nonempty tokens plus generation must fit the context bound (at most 512)".into(),
            );
        }
        tokens
            .iter()
            .map(|&v| {
                if v >= 0 && (v as u64) < self.cfg.vocab_size as u64 {
                    Ok(v as u32)
                } else {
                    Err("token ID is outside the vocabulary".into())
                }
            })
            .collect()
    }

    fn with_forward<R>(
        &mut self,
        f: impl FnOnce(&mut dyn FnMut(u32, usize) -> vitni_tensor::Result<Tensor>) -> Result<R>,
    ) -> Result<R> {
        if !super::environment_ok() {
            return Err("unsupported CPU floating-point environment".into());
        }
        let cfg = self.cfg;
        match &mut self.backend {
            Backend::Cpu { blob, state } => {
                let gguf = GgufFile::parse(blob).map_err(|e| format!("GGUF view: {e:?}"))?;
                let weights = QuantizedWeights::from_gguf(&gguf, &cfg)
                    .map_err(|e| format!("weight views: {e:?}"))?;
                state.key_cache_mut().fill(0.0);
                state.value_cache_mut().fill(0.0);
                f(&mut |token, pos| forward_quantized::step(&cfg, &weights, state, token, pos))
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal(engine) => metal_pool(|| {
                engine.reset();
                f(&mut |token, pos| {
                    Tensor::from_f32(engine.step(token, pos), Shape::new(&[cfg.vocab_size])?)
                })
            }),
        }
    }

    pub fn forward(&mut self, tokens: &[i64], out: &mut [f32]) -> Result<()> {
        let tokens = self.validate(tokens, 0)?;
        let vocab = self.cfg.vocab_size;
        if tokens.len().checked_mul(vocab) != Some(out.len()) {
            return Err("wrong output buffer extent".into());
        }
        self.with_forward(|step| {
            for (pos, &token) in tokens.iter().enumerate() {
                let logits = step(token, pos).map_err(|e| format!("forward: {e:?}"))?;
                let Storage::Cpu(values) = logits.storage() else {
                    return Err("expected host logits from engine".into());
                };
                let row = values.as_f32_slice();
                if row.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite model logits".into());
                }
                out[pos * vocab..(pos + 1) * vocab].copy_from_slice(row);
            }
            Ok(())
        })?;
        self.forward_calls += 1;
        Ok(())
    }

    pub fn generate(&mut self, tokens: &[i64], n: usize, model_id: &str) -> Result<String> {
        let tokens = self.validate(tokens, n)?;
        let cfg = self.cfg;
        let hash = self.weights_hash;
        let req = inference::Request {
            model_id,
            prompt_tokens: &tokens,
            n_new_tokens: n,
        };
        let outcome = self.with_forward(|step| {
            inference::run_quantized_with_forward(
                &cfg,
                &hash,
                &req,
                &mut vitni_tensor::cert::NullSink,
                0,
                step,
            )
            .map(|(o, _)| o)
            .map_err(|e| format!("inference: {e:?}"))
        })?;
        self.generate_calls += 1;
        let c = &outcome.cert;
        let v1 = vitni_tensor::cert::builder::compute_digest_v1(
            &c.inputs,
            &c.outputs,
            &c.ops,
            &c.activations,
            &c.interventions,
        );
        let tokens = outcome
            .generated_tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!("{{\"model_digest\":\"{}\",\"regime\":\"{}\",\"model_digest_v1\":\"{}\",\"weights_hash\":\"{}\",\"tokens\":[{}]}}",
            hex(&c.digest),vitni_tensor::cert::builder::REGIME,hex(&v1),hex(&hash),tokens))
    }

    pub fn info(&self) -> String {
        let (backend,flags)=match &self.backend {
            Backend::Cpu {..}=>("cpu","\"weight_views\":\"rebuilt per request without matrix copies\"".to_string()),
            #[cfg(all(feature="metal", target_os="macos"))]
            Backend::Metal(m)=>("metal",format!("\"q4_variant\":\"{}\",\"rms_variant\":\"{}\",\"q6_cooperative\":{},\"attention_cooperative\":{},\"concurrent_dispatch\":{},\"q4_storage\":{:?}",
                m.q4k_variant(),m.rms_variant(),m.q6k_cooperative_enabled(),m.attention_cooperative_enabled(),m.concurrent_dispatch_enabled(),m.q4k_metadata_storage())),
        };
        // A tuple is formatted as an array for JSON, without an external serializer.
        let flags = flags.replace('(', "[").replace(')', "]");
        format!("{{\"backend\":\"{backend}\",\"optimized\":{},\"vocab_size\":{},\"model_context\":{},\"max_context\":{},\"weights_hash\":\"{}\",\"model_loads\":1,\"forward_calls\":{},\"generate_calls\":{},{} }}",
            self.optimized,self.cfg.vocab_size,self.cfg.seq_len,self.cfg.seq_len.min(512),hex(&self.weights_hash),self.forward_calls,self.generate_calls,flags)
    }
}

type Shared = Arc<Mutex<Session>>;
static REGISTRY: OnceLock<Mutex<HashMap<u64, Shared>>> = OnceLock::new();
static NEXT: AtomicU64 = AtomicU64::new(1);
thread_local! {static ERROR:RefCell<CString>=RefCell::new(CString::new("").unwrap());}
fn registry() -> &'static Mutex<HashMap<u64, Shared>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}
// Command buffers/encoders are autoreleased Objective-C objects. A resident
// non-Cocoa process must drain them between requests, including error paths.
fn metal_pool<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        objc::rc::autoreleasepool(f)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        f()
    }
}

fn ffi(f: impl FnOnce() -> Result<()> + std::panic::UnwindSafe) -> i32 {
    match metal_pool(|| catch_unwind(f)) {
        Ok(Ok(())) => 0,
        failure => {
            let message = match failure {
                Ok(Err(e)) => e,
                _ => "native model panic; close and reopen the model".into(),
            };
            ERROR.with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            1
        }
    }
}
fn session(id: u64) -> Result<Shared> {
    registry()
        .lock()
        .map_err(|_| "model registry poisoned")?
        .get(&id)
        .cloned()
        .ok_or_else(|| "model is closed or handle is invalid".into())
}
unsafe fn text<'a>(p: *const u8, n: usize) -> Result<&'a str> {
    if n > 1024 * 1024 {
        return Err("string too long".into());
    }
    let bytes = unsafe { super::input(p, n) }.ok_or("invalid string buffer")?;
    std::str::from_utf8(bytes).map_err(|_| "expected UTF-8".into())
}
unsafe fn tokens<'a>(p: *const i64, n: usize) -> Result<&'a [i64]> {
    if n > 512 {
        return Err("tokens exceed 512 context bound".into());
    }
    unsafe { super::input(p, n) }.ok_or_else(|| "invalid token buffer".into())
}
unsafe fn json_out(value: &str, out: *mut u8, cap: usize, written: *mut usize) -> Result<()> {
    if out.is_null() || written.is_null() || cap < value.len() {
        return Err("JSON output buffer is too small".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(value.as_ptr(), out, value.len());
        *written = value.len();
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn vitni_model_last_error() -> *const std::ffi::c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}

/// FFI callers own valid aligned buffers for all stated lengths throughout calls.
#[no_mangle]
pub unsafe extern "C" fn vitni_model_open(
    path: *const u8,
    len: usize,
    backend: u32,
    optimized: u32,
    out: *mut u64,
) -> i32 {
    ffi(|| {
        if out.is_null() {
            return Err("null model handle output".into());
        }
        let s = Session::open(unsafe { text(path, len) }?, backend, optimized != 0)?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        registry()
            .lock()
            .map_err(|_| "model registry poisoned")?
            .insert(id, Arc::new(Mutex::new(s)));
        unsafe {
            *out = id;
        }
        Ok(())
    })
}
#[no_mangle]
pub extern "C" fn vitni_model_close(id: u64) -> i32 {
    ffi(|| {
        registry()
            .lock()
            .map_err(|_| "model registry poisoned")?
            .remove(&id)
            .ok_or("model is already closed")?;
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn vitni_model_info(
    id: u64,
    out: *mut u8,
    cap: usize,
    written: *mut usize,
) -> i32 {
    ffi(|| {
        let s = session(id)?;
        let s = s.lock().map_err(|_| "model state poisoned")?;
        unsafe { json_out(&s.info(), out, cap, written) }
    })
}
#[no_mangle]
pub unsafe extern "C" fn vitni_model_forward(
    id: u64,
    p: *const i64,
    n: usize,
    out: *mut f32,
    out_len: usize,
    seconds: *mut f64,
) -> i32 {
    ffi(|| {
        let s = session(id)?;
        let mut s = s.lock().map_err(|_| "model state poisoned")?;
        let tok = unsafe { tokens(p, n) }?;
        s.validate(tok, 0)?;
        if out.is_null()
            || out as usize % 4 != 0
            || seconds.is_null()
            || n.checked_mul(s.cfg.vocab_size) != Some(out_len)
            || out_len > isize::MAX as usize / 4
        {
            return Err("invalid model output buffer".into());
        }
        let start = Instant::now();
        s.forward(tok, unsafe { std::slice::from_raw_parts_mut(out, out_len) })?;
        unsafe {
            *seconds = start.elapsed().as_secs_f64();
        }
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn vitni_model_generate(
    id: u64,
    p: *const i64,
    n: usize,
    extra: usize,
    model_id: *const u8,
    id_len: usize,
    out: *mut u8,
    cap: usize,
    written: *mut usize,
    seconds: *mut f64,
) -> i32 {
    ffi(|| {
        if seconds.is_null() || cap < 16384 {
            return Err("invalid generation output buffer".into());
        }
        let s = session(id)?;
        let mut s = s.lock().map_err(|_| "model state poisoned")?;
        let start = Instant::now();
        let result = s.generate(unsafe { tokens(p, n) }?, extra, unsafe {
            text(model_id, id_len)
        }?)?;
        unsafe {
            json_out(&result, out, cap, written)?;
            *seconds = start.elapsed().as_secs_f64();
        }
        Ok(())
    })
}
