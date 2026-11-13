//! Differential checks for the experimental unchanged-regime Q4_K kernel.
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use std::{ffi::c_void, time::Instant};
use vitni_tensor::ops::quant::linear_q4_k_fused;

const SOURCE: &str = include_str!("../../kernels/q4k.metal");

struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: Vec<metal::ComputePipelineState>,
}

impl Gpu {
    fn new() -> Self {
        let device =
            Device::system_default().expect("Metal GPU is required; do not count a skip as a pass");
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SOURCE, &options)
            .expect("compile Q4_K kernels");
        let pipelines = [
            "q4k_linear",
            "q4k_linear_acc",
            "q4k_linear_cooperative",
            "q4k_linear_cooperative_acc",
        ]
        .iter()
        .map(|name| {
            let function = library.get_function(name, None).unwrap();
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap()
        })
        .collect();
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            pipelines,
        }
    }

    fn run(
        &self,
        x: &[f32],
        weights: &[u8],
        initial: &[f32],
        m: usize,
        cooperative: bool,
        accumulate: bool,
        repetitions: usize,
    ) -> (Vec<f32>, f64) {
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer =
            |p: *const c_void, n: usize| self.device.new_buffer_with_data(p, n as u64, shared);
        let xb = buffer(x.as_ptr().cast(), x.len() * 4);
        let wb = buffer(weights.as_ptr().cast(), weights.len());
        // Guard elements detect an accidental store from a padded output row.
        let yb = buffer(initial.as_ptr().cast(), initial.len() * 4);
        let dims = [x.len() as u32, m as u32, (x.len() / 256) as u32];
        let db = buffer(dims.as_ptr().cast(), 12);
        let pipeline = &self.pipelines[usize::from(cooperative) * 2 + usize::from(accumulate)];
        assert!(pipeline.max_total_threads_per_threadgroup() >= 64);
        let threads = m * if cooperative { 8 } else { 1 };
        let mut elapsed = 0.0;
        for _ in 0..repetitions {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    initial.as_ptr(),
                    yb.contents().cast(),
                    initial.len(),
                );
            }
            let start = Instant::now();
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (index, b) in [&xb, &wb, &yb, &db].iter().enumerate() {
                encoder.set_buffer(index as u64, Some(b), 0);
            }
            encoder.dispatch_thread_groups(
                MTLSize::new(threads.div_ceil(64) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            elapsed += start.elapsed().as_secs_f64();
            assert_eq!(
                command.status(),
                MTLCommandBufferStatus::Completed,
                "GPU command failed"
            );
        }
        let values =
            unsafe { std::slice::from_raw_parts(yb.contents().cast::<f32>(), initial.len()) }
                .to_vec();
        (values, elapsed / repetitions as f64)
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
            let r = ((next() >> 8) as f32 / 16777216.0) * 2.0 - 1.0;
            match i % 19 {
                0 => 0.0,
                1 => -0.0,
                2 => r * 1.0e4,
                3 => r * 1.0e-4,
                _ => r,
            }
        })
        .collect();
    let mut weights: Vec<u8> = (0..m * (k / 256) * 144).map(|_| next() as u8).collect();
    // Finite, ordinary Q4_K scale encodings; vary exponent and mantissa.
    for b in weights.chunks_exact_mut(144) {
        let d = 0x3000u16 + (next() as u16 & 0x0fff);
        let minimum = 0x2800u16 + (next() as u16 & 0x0fff);
        b[..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&minimum.to_le_bytes());
    }
    let initial = (0..m + 3)
        .map(|i| {
            if i < m {
                (i as f32 - 17.0) * 0.375
            } else {
                123456.75
            }
        })
        .collect();
    (x, weights, initial)
}

fn assert_bits(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (index, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: index {index}, {x} vs {y}"
        );
    }
}

#[test]
#[ignore = "requires access to a real Metal GPU"]
fn cooperative_q4k_matches_cpu_at_chunk_and_row_boundaries() {
    let gpu = Gpu::new();
    let shapes = [
        (256, 1),
        (768, 7),
        (8192, 8),
        (8448, 9),
        (14336, 33),
        (16640, 9),
        (27648, 65),
    ];
    let mut checked = 0usize;
    for (k, m) in shapes {
        for seed in [7, 0x12345678, 0xfedcba9876543210] {
            let (x, weights, initial) = fixture(k, m, seed);
            let mut reference = vec![0.0; m];
            linear_q4_k_fused(&x, &weights, &mut reference, 1, k, m).unwrap();
            for accumulate in [false, true] {
                let mut expected = initial.clone();
                for i in 0..m {
                    expected[i] = if accumulate {
                        initial[i] + reference[i]
                    } else {
                        reference[i]
                    };
                }
                for cooperative in [false, true] {
                    let (actual, _) =
                        gpu.run(&x, &weights, &initial, m, cooperative, accumulate, 1);
                    assert_bits(
                        &actual,
                        &expected,
                        &format!(
                            "K={k} M={m} seed={seed} cooperative={cooperative} acc={accumulate}"
                        ),
                    );
                    checked += m;
                }
            }
        }
    }
    eprintln!("checked {checked} scalar outputs against the CPU, plus padded-row guards");
}

#[test]
#[ignore = "diagnostic benchmark; requires a real Metal GPU"]
fn cooperative_q4k_dispatch_benchmark() {
    let gpu = Gpu::new();
    for (k, m) in [(2048, 2048), (2048, 5632), (5632, 2048), (14336, 4096)] {
        let (x, weights, initial) = fixture(k, m, 42);
        let (reference, _) = gpu.run(&x, &weights, &initial, m, false, false, 2);
        let (candidate, _) = gpu.run(&x, &weights, &initial, m, true, false, 2);
        assert_bits(&candidate, &reference, "benchmark output");
        // Alternate order to reduce a simple warmup/order bias. These host wall
        // measurements include command encoding, dispatch and waiting.
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..6 {
            for cooperative in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let (_, seconds) = gpu.run(&x, &weights, &initial, m, cooperative, false, 16);
                times[usize::from(cooperative)].push(seconds);
            }
        }
        for t in &mut times {
            t.sort_by(|a, b| a.total_cmp(b));
        }
        let base = (times[0][2] + times[0][3]) / 2.0;
        let candidate = (times[1][2] + times[1][3]) / 2.0;
        println!(
            "Q4K_BENCH K={k} M={m} baseline_us={:.3} cooperative_us={:.3} speedup={:.4}",
            base * 1e6,
            candidate * 1e6,
            base / candidate
        );
    }
}
