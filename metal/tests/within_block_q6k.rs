//! Research within-block Q6: retained cooperative controls versus CPU and four-lane blocks.
use metal::objc::{msg_send, sel, sel_impl};
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use std::{ffi::c_void, time::Instant};
use vitni_tensor::ops::quant::quantize_block_q8;
const SOURCE: &str = include_str!("../../kernels/q6k_int.metal");
const CANDIDATE: &str = include_str!("fixtures/q6_within_block.metal");
pub struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    quant: metal::ComputePipelineState,
    dots: Vec<metal::ComputePipelineState>,
}
pub fn assert_bits(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label} output={i} {x} vs {y}");
    }
}
impl Gpu {
    pub fn new() -> Self {
        let device = Device::system_default().expect("real Metal GPU required");
        let o = CompileOptions::new();
        o.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&format!("{SOURCE}\n{CANDIDATE}"), &o)
            .expect("compile Q6 kernels");
        let pl = |name: &str| {
            device
                .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
                .unwrap()
        };
        let quant = pl("q6k_quantize");
        let dots = [
            "q6k_integer_dot_cooperative",
            "q6k_integer_dot_cooperative_acc",
            "q6k_integer_dot_within_block",
            "q6k_integer_dot_within_block_acc",
        ]
        .iter()
        .map(|n| pl(n))
        .collect::<Vec<_>>();
        assert_eq!(dots[2].thread_execution_width(), 32);
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            quant,
            dots,
        }
    }
    pub fn run(
        &self,
        x: &[f32],
        w: &[u8],
        initial: &[f32],
        m: usize,
        cooperative: bool,
        acc: bool,
        reps: usize,
    ) -> (Vec<f32>, f64, f64) {
        let n = x.len() / 256;
        assert!(n >= 1 && n <= 64);
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer =
            |p: *const c_void, len: usize| self.device.new_buffer_with_data(p, len as u64, shared);
        let xb = buffer(x.as_ptr().cast(), x.len() * 4);
        let wb = buffer(w.as_ptr().cast(), w.len());
        let yb = buffer(initial.as_ptr().cast(), initial.len() * 4);
        let dx = self.device.new_buffer((n * 4) as u64, shared);
        let qs = self.device.new_buffer((n * 256) as u64, shared);
        let dims = [x.len() as u32, m as u32, n as u32];
        let db = buffer(dims.as_ptr().cast(), 12);
        let ns = n as u32;
        let factor = if cooperative { 32 } else { 8 };
        let dot = &self.dots[usize::from(cooperative) * 2 + usize::from(acc)];
        let start = Instant::now();
        let c = self.queue.new_command_buffer();
        let e = c.new_compute_command_encoder();
        for _ in 0..reps {
            e.set_compute_pipeline_state(&self.quant);
            e.set_buffer(0, Some(&xb), 0);
            e.set_buffer(1, Some(&dx), 0);
            e.set_buffer(2, Some(&qs), 0);
            e.set_bytes(3, 4, (&ns as *const u32).cast());
            e.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(64) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            e.memory_barrier_with_resources(&[&dx, &qs]);
            e.set_compute_pipeline_state(dot);
            for (i, b) in [&dx, &qs, &wb, &yb, &db].iter().enumerate() {
                e.set_buffer(i as u64, Some(b), 0);
            }
            e.dispatch_thread_groups(
                MTLSize::new((m * factor).div_ceil(64) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            e.memory_barrier_with_resources(&[&yb, &dx, &qs]);
        }
        e.end_encoding();
        c.commit();
        c.wait_until_completed();
        let host = start.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(c.status(), MTLCommandBufferStatus::Completed);
        let a: f64 = unsafe { msg_send![c, GPUStartTime] };
        let b: f64 = unsafe { msg_send![c, GPUEndTime] };
        assert!(b > a && a > 0.);
        let ds = unsafe { std::slice::from_raw_parts(dx.contents().cast::<f32>(), n) };
        let is = unsafe { std::slice::from_raw_parts(qs.contents().cast::<i8>(), n * 256) };
        for s in 0..n {
            let q = quantize_block_q8(&x[s * 256..(s + 1) * 256]);
            assert_eq!(
                ds[s].to_bits(),
                q.d.to_bits(),
                "unchanged quant scale block={s}"
            );
            assert_eq!(
                &is[s * 256..(s + 1) * 256],
                &q.qs,
                "unchanged quant integers"
            );
        }
        let out = unsafe { std::slice::from_raw_parts(yb.contents().cast::<f32>(), initial.len()) }
            .to_vec();
        (out, host, (b - a) / reps as f64)
    }
}

fn fixture(k: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    let x = (0..k)
        .map(|i| {
            let a = ((next() >> 8) as f32 / 16777216.) * 2. - 1.;
            if seed == 7 {
                if i % 2 == 0 {
                    0.
                } else {
                    -0.
                }
            } else {
                match i % 19 {
                    0 => 0.,
                    1 => -0.,
                    2 => a * 1.0e4,
                    3 => a * 1.0e-4,
                    _ => a,
                }
            }
        })
        .collect();
    let mut w: Vec<u8> = (0..m * k / 256 * 210).map(|_| next() as u8).collect();
    for b in w.chunks_exact_mut(210) {
        let bits =
            (0x1000u16 + (next() as u16 & 0x2fff)) | if next() & 1 != 0 { 0x8000 } else { 0 };
        b[208..210].copy_from_slice(&bits.to_le_bytes());
    }
    let y = (0..m + 3)
        .map(|i| {
            if i < m {
                (i as f32 - 17.) * 0.375
            } else {
                123456.75
            }
        })
        .collect();
    (x, w, y)
}
#[test]
#[ignore = "requires an actual Metal GPU"]
fn within_block_q6_matches_cpu_at_block_and_row_boundaries() {
    let gpu = Gpu::new();
    let mut count = 0;
    for (ns, m) in [
        (1, 1),
        (2, 7),
        (3, 8),
        (7, 9),
        (8, 33),
        (9, 65),
        (22, 9),
        (31, 7),
        (32, 8),
        (33, 9),
        (56, 33),
        (63, 7),
        (64, 9),
    ] {
        for seed in [7, 0x12345678, 0xfedcba9876543210] {
            let k = ns * 256;
            let (x, w, initial) = fixture(k, m, seed);
            let mut cpu = vec![0.; m];
            vitni_tensor::ops::quant::linear_q6_k_integer(&x, &w, &mut cpu, 1, k, m).unwrap();
            for coop in [false, true] {
                for acc in [false, true] {
                    let mut expected = initial.clone();
                    for i in 0..m {
                        expected[i] = if acc { initial[i] + cpu[i] } else { cpu[i] };
                    }
                    let (actual, _, _) = gpu.run(&x, &w, &initial, m, coop, acc, 1);
                    assert_bits(
                        &actual,
                        &expected,
                        &format!("ns={ns} m={m} seed={seed} coop={coop} acc={acc}"),
                    );
                    count += m;
                }
            }
        }
    }
    eprintln!("{count} scalar comparisons including current controls; quantization and output guards passed");
}
fn reference_tree(mut leaves: Vec<String>) -> String {
    while leaves.len() > 1 {
        leaves = leaves
            .chunks(2)
            .map(|p| {
                if p.len() == 2 {
                    format!("({}+{})", p[0], p[1])
                } else {
                    p[0].clone()
                }
            })
            .collect();
    }
    leaves[0].clone()
}
#[test]
fn within_block_float_tree_has_identical_topology() {
    for n in 1..=64 {
        let leaves: Vec<_> = (0..n).map(|i| i.to_string()).collect();
        let a = reference_tree(leaves.clone());
        let b = reference_tree(
            leaves
                .chunks(8)
                .map(|c| reference_tree(c.to_vec()))
                .collect(),
        );
        assert_eq!(a, b, "nsuper={n}");
    }
}

#[test]
#[ignore = "requires an actual Metal GPU"]
fn within_block_q6_extreme_integer_partials_match_cpu() {
    // Upper bound also covers any grouping of the scaled signed products.
    let bound = 256u64 * 32 * 127 * 128;
    assert_eq!(bound, 133_169_152);
    assert!(bound < i32::MAX as u64);
    let gpu = Gpu::new();
    let mut count = 0;
    for ns in [1, 3, 8, 9, 22, 63, 64] {
        for case in 0..3 {
            let k = ns * 256;
            let m = 9;
            let x: Vec<f32> = (0..k)
                .map(|i| {
                    if case == 1 || (case == 2 && i % 2 == 1) {
                        -127.0
                    } else {
                        127.0
                    }
                })
                .collect();
            let mut w = vec![0u8; m * ns * 210];
            for block in w.chunks_exact_mut(210) {
                if case == 1 {
                    block[..192].fill(255); // Every Q6 integer is +31.
                    block[192..208].fill(127);
                } else {
                    // All Q6 integers are -32; -128 scales hit the positive bound.
                    block[192..208].fill(128);
                    if case == 2 {
                        for j in (193..208).step_by(2) {
                            block[j] = 127;
                        }
                    }
                }
                block[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
            let initial: Vec<_> = (0..m + 3)
                .map(|i| if i < m { i as f32 - 4.0 } else { 123456.75 })
                .collect();
            let mut cpu = vec![0.0; m];
            vitni_tensor::ops::quant::linear_q6_k_integer(&x, &w, &mut cpu, 1, k, m).unwrap();
            for candidate in [false, true] {
                for acc in [false, true] {
                    let mut expected = initial.clone();
                    for i in 0..m {
                        expected[i] = if acc { initial[i] + cpu[i] } else { cpu[i] };
                    }
                    let (actual, _, _) = gpu.run(&x, &w, &initial, m, candidate, acc, 1);
                    assert_bits(
                        &actual,
                        &expected,
                        &format!("ns={ns} case={case} candidate={candidate} acc={acc}"),
                    );
                    count += m;
                }
            }
        }
    }
    eprintln!("{count} extreme scalar comparisons plus quantization and output guards passed");
}
