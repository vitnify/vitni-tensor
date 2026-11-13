//! CPU-reference attention checks for grouped queries, tails and context limits.
use metal::objc::{msg_send, sel, sel_impl};
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use std::{ffi::c_void, time::Instant};
use vitni_tensor::{ops::quant, Shape, Storage, Tensor};
const SOURCE: &str = include_str!("../../kernels/forward_ops.metal");
pub struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: Vec<metal::ComputePipelineState>,
}
pub fn assert_bits(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label} index={i} {x} vs {y}");
    }
}
impl Gpu {
    pub fn new() -> Self {
        let device = Device::system_default().expect("real Metal GPU required");
        let o = CompileOptions::new();
        o.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(SOURCE, &o)
            .expect("compile attention shaders");
        let pipelines = ["attention", "attention_cooperative"]
            .iter()
            .map(|n| {
                device
                    .new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(pipelines[1].max_total_threads_per_threadgroup() >= 64);
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            pipelines,
        }
    }
    pub fn run(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        dims: [u32; 5],
        cooperative: bool,
        reps: usize,
    ) -> (Vec<f32>, f64, f64) {
        let count = (dims[0] * dims[1]) as usize;
        let initial = vec![123456.75f32; count + 3];
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer =
            |p: *const c_void, n: usize| self.device.new_buffer_with_data(p, n as u64, shared);
        let qb = buffer(q.as_ptr().cast(), q.len() * 4);
        let kb = buffer(k.as_ptr().cast(), k.len() * 4);
        let vb = buffer(v.as_ptr().cast(), v.len() * 4);
        let out = buffer(initial.as_ptr().cast(), initial.len() * 4);
        let db = buffer(dims.as_ptr().cast(), 20);
        let threads = dims[0] as usize * if cooperative { 64 } else { 1 };
        let start = Instant::now();
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines[usize::from(cooperative)]);
        for (i, b) in [&qb, &kb, &vb, &out, &db].iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for _ in 0..reps {
            enc.dispatch_thread_groups(
                MTLSize::new(threads.div_ceil(64) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            enc.memory_barrier_with_resources(&[&out]);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let host = start.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
        let a: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let b: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        assert!(b > a && a > 0.);
        let y = unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), initial.len()) }
            .to_vec();
        assert_bits(&y[count..], &initial[count..], "output guards");
        (y[..count].to_vec(), host, (b - a) / reps as f64)
    }
}
fn attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    nh: usize,
    hs: usize,
    kvd: usize,
    km: usize,
    tlen: usize,
) -> Vec<f32> {
    let mut out = vec![0.; nh * hs];
    let sqrt = libm::sqrtf(hs as f32);
    for h in 0..nh {
        let kh = h / km;
        let mut scores: Vec<f32> = (0..tlen)
            .map(|t| {
                quant::canonical_dot_pub(
                    &q[h * hs..(h + 1) * hs],
                    &k[t * kvd + kh * hs..t * kvd + (kh + 1) * hs],
                    hs,
                ) / sqrt
            })
            .collect();
        let st = Tensor::from_f32(scores, Shape::new(&[tlen]).unwrap()).unwrap();
        scores = {
            let out = st.softmax_last_dim().unwrap();
            let Storage::Cpu(b) = out.storage() else {
                panic!("non CPU")
            };
            b.as_f32_slice().to_vec()
        };
        for d in 0..hs {
            let vc: Vec<f32> = (0..tlen).map(|t| v[t * kvd + kh * hs + d]).collect();
            out[h * hs + d] = quant::canonical_dot_pub(&scores, &vc, tlen);
        }
    }
    out
}

fn fixture(nh: usize, hs: usize, km: usize, t: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / 16777216.) * 2. - 1.
    };
    let mut make = |n: usize, v: bool| -> Vec<f32> {
        (0..n)
            .map(|i| {
                if seed == 7 {
                    if i % 2 == 0 {
                        0.
                    } else {
                        -0.
                    }
                } else {
                    next() * if v && i % 19 == 2 { 1e4 } else { 1. }
                }
            })
            .collect()
    };
    let q = make(nh * hs, false);
    let k = make(t * (nh / km) * hs, false);
    let v = make(t * (nh / km) * hs, true);
    (q, k, v)
}
#[test]
#[ignore = "requires actual Metal GPU"]
fn cooperative_attention_matches_cpu_for_gqa_tails_and_context_boundaries() {
    let gpu = Gpu::new();
    let mut checked = 0usize;
    for (nh, hs, km) in [
        (1, 1, 1),
        (3, 7, 1),
        (4, 31, 2),
        (6, 48, 1),
        (32, 64, 8),
        (3, 65, 1),
        (8, 128, 4),
    ] {
        for t in [1, 2, 7, 8, 9, 31, 32, 63, 64, 65, 127, 128, 511, 512] {
            for seed in [7, 0x12345678, 0xfedcba9876543210] {
                let (q, k, v) = fixture(nh, hs, km, t, seed);
                let kvd = nh / km * hs;
                let expected = attention_cpu(&q, &k, &v, nh, hs, kvd, km, t);
                for cooperative in [false, true] {
                    let (actual, _, _) = gpu.run(
                        &q,
                        &k,
                        &v,
                        [nh as u32, hs as u32, kvd as u32, km as u32, (t - 1) as u32],
                        cooperative,
                        1,
                    );
                    assert_bits(
                        &actual,
                        &expected,
                        &format!("nh={nh} hs={hs} km={km} t={t} seed={seed} coop={cooperative}"),
                    );
                    checked += expected.len();
                }
            }
        }
    }
    // Upper single-chunk head-size boundary, without a large stress workload.
    for t in [1, 7] {
        let (q, k, v) = fixture(1, 8192, 1, t, 42);
        let expected = attention_cpu(&q, &k, &v, 1, 8192, 8192, 1, t);
        for c in [false, true] {
            let (a, _, _) = gpu.run(&q, &k, &v, [1, 8192, 8192, 1, (t - 1) as u32], c, 1);
            assert_bits(&a, &expected, "head_size=8192");
            checked += a.len();
        }
    }
    eprintln!("{checked} scalar CPU comparisons including current controls; output guards passed");
}
