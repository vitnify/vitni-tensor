//! Independent CPU checks for separate dispatch and normalization-fusion trials.
#[path = "support/dispatch_rms_gpu.rs"]
mod gpu;
#[path = "../src/q4_metadata.rs"]
mod metadata;
use gpu::{check, Gpu};
use vitni_tensor::{Shape, Storage, Tensor};

fn fixture(k: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    let x = (0..k)
        .map(|i| {
            let v = (next() as i32 as f32) / i32::MAX as f32;
            if seed == 0 {
                if i % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            } else {
                match i % 19 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => v * 1e4,
                    3 => v * 1e-4,
                    _ => v,
                }
            }
        })
        .collect();
    let rw = (0..k)
        .map(|i| {
            let v = (next() as i32 as f32) / i32::MAX as f32;
            match i % 9 {
                0 => -0.0,
                1 => 0.0,
                _ => v,
            }
        })
        .collect();
    let mut w: Vec<_> = (0..m * k / 256 * 144).map(|_| next() as u8).collect();
    for b in w.chunks_exact_mut(144) {
        let cases = [
            0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3000, 0x3800,
        ];
        let h = cases[next() as usize % cases.len()] as u16;
        b[..2].copy_from_slice(&h.to_le_bytes());
        b[2..4].copy_from_slice(&0x3000u16.to_le_bytes());
    }
    (x, rw, w)
}
fn initial(m: usize) -> Vec<f32> {
    (0..m + 3)
        .map(|i| if i < m { i as f32 - 17.0 } else { 123456.75 })
        .collect()
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn concurrent_independent_q4_pairs_match_cpu() {
    let gpu = Gpu::new();
    let mut words = 0;
    for (k, m) in [(256, 1), (768, 7), (2048, 9), (4096, 33)] {
        for seed in [0, 7, 0x12345678] {
            let (x, _, w) = fixture(k, m, seed);
            let (_, _, w2) = fixture(k, m, seed + 11);
            let mut ys = [vec![0.0; m], vec![0.0; m]];
            for (i, weights) in [&w, &w2].iter().enumerate() {
                vitni_tensor::ops::quant::linear_q4_k_fused(&x, weights, &mut ys[i], 1, k, m)
                    .unwrap();
            }
            let xb = gpu.floats(&x);
            let w0 = gpu.buffer(&metadata::pack(&w));
            let w1 = gpu.buffer(&metadata::pack(&w2));
            let init = initial(m);
            for concurrent in [false, true] {
                let (a, b, _, _) = gpu.pair(&xb, [&w0, &w1], &init, k, m, concurrent, 32);
                for (actual, y) in [&a, &b].iter().zip(&ys) {
                    let mut expected = init.clone();
                    expected[..m].copy_from_slice(y);
                    check(actual, &expected, "concurrent Q4 and guards");
                    words += m;
                }
            }
        }
    }
    eprintln!("{words} concurrent/serial scalar comparisons plus guards passed");
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn rms_to_q4_fusion_matches_cpu_at_boundaries() {
    let gpu = Gpu::new();
    let mut words = 0;
    for (k, m) in [(256, 1), (768, 7), (2048, 9), (4096, 33)] {
        for seed in [0, 7, 0x12345678] {
            let (x, rw, w) = fixture(k, m, seed);
            let tx = Tensor::from_f32(x.clone(), Shape::new(&[k]).unwrap()).unwrap();
            let tw = Tensor::from_f32(rw.clone(), Shape::new(&[k]).unwrap()).unwrap();
            let n = tx.rms_norm(&tw, 1e-5).unwrap();
            let Storage::Cpu(s) = n.storage() else {
                panic!("CPU required")
            };
            let mut y = vec![0.0; m];
            vitni_tensor::ops::quant::linear_q4_k_fused(s.as_f32_slice(), &w, &mut y, 1, k, m)
                .unwrap();
            let xb = gpu.floats(&x);
            let rb = gpu.floats(&rw);
            let wb = gpu.buffer(&metadata::pack(&w));
            let init = initial(m);
            for fused in [false, true] {
                for acc in [false, true] {
                    let (a, _, _) = gpu.fusion(&xb, &rb, &wb, &init, k, m, 1e-5, fused, acc, 1);
                    let mut expected = init.clone();
                    for i in 0..m {
                        expected[i] = if acc { init[i] + y[i] } else { y[i] };
                    }
                    check(&a, &expected, "CPU RMS+Q4 and guards");
                    words += m;
                }
            }
        }
    }
    eprintln!("{words} fused/unfused scalar comparisons plus guards passed");
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn concurrent_q6_scratch_reuse_matches_cpu() {
    use metal::{
        CompileOptions, Device, MTLCommandBufferStatus, MTLDispatchType, MTLResourceOptions,
        MTLSize,
    };
    let d = Device::system_default().expect("real GPU required");
    let queue = d.new_command_queue();
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let lib = d
        .new_library_with_source(include_str!("../../kernels/q6k_int.metal"), &opts)
        .unwrap();
    let pl = |name: &str| {
        d.new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
            .unwrap()
    };
    let quant = pl("q6k_quantize");
    let dot = pl("q6k_integer_dot_cooperative");
    let k = 768;
    let m = 65;
    let n = 3u32;
    let xs: [Vec<f32>; 2] = [0, 1].map(|v| {
        (0..k)
            .map(|i| ((i as f32 * 0.17) + v as f32 * 13.0).sin() * 127.0)
            .collect()
    });
    let mut w: Vec<u8> = (0..m * 3 * 210).map(|i| (i * 73 + 19) as u8).collect();
    for b in w.chunks_exact_mut(210) {
        b[208..].copy_from_slice(&0x3800u16.to_le_bytes());
    }
    let shared = MTLResourceOptions::StorageModeShared;
    let buf =
        |bytes: &[u8]| d.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, shared);
    let fb = |v: &[f32]| buf(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>());
    let xb = xs.each_ref().map(|v| fb(v));
    let wb = buf(&w);
    let dx = d.new_buffer(12, shared);
    let qs = d.new_buffer(768, shared);
    let init = initial(m);
    let ys = [fb(&init), fb(&init)];
    let dims = [k as u32, m as u32, n];
    let mut expected = [init.clone(), init.clone()];
    for i in 0..2 {
        vitni_tensor::ops::quant::linear_q6_k_integer(&xs[i], &w, &mut expected[i][..m], 1, k, m)
            .unwrap();
    }
    let c = queue.new_command_buffer();
    let e = c.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    for _ in 0..32 {
        for i in 0..2 {
            // WAR: earlier dots must finish reading before quantization overwrites.
            e.memory_barrier_with_resources(&[&dx, &qs]);
            e.set_compute_pipeline_state(&quant);
            for (j, b) in [&xb[i], &dx, &qs].iter().enumerate() {
                e.set_buffer(j as u64, Some(b), 0);
            }
            e.set_bytes(3, 4, (&n as *const u32).cast());
            e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
            e.memory_barrier_with_resources(&[&dx, &qs]);
            e.set_compute_pipeline_state(&dot);
            for (j, b) in [&dx, &qs, &wb, &ys[i]].iter().enumerate() {
                e.set_buffer(j as u64, Some(b), 0);
            }
            e.set_bytes(4, 12, dims.as_ptr().cast());
            e.dispatch_thread_groups(
                MTLSize::new((m * 8).div_ceil(64) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
        }
        e.memory_barrier_with_resources(&[&ys[0], &ys[1]]);
    }
    e.end_encoding();
    c.commit();
    c.wait_until_completed();
    assert_eq!(c.status(), MTLCommandBufferStatus::Completed);
    for i in 0..2 {
        let actual = unsafe { std::slice::from_raw_parts(ys[i].contents().cast::<f32>(), m + 3) };
        check(actual, &expected[i], "Q6 scratch lifetime and guards");
    }
    eprintln!("130 Q6 scratch-reuse scalar outputs plus guards passed after64 projections");
}
