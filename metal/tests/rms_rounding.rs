//! Actual CPU RMS versus both Metal implementations, including reduction tails.
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use vitni_tensor::{storage::Storage, Shape, Tensor};

struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: Vec<metal::ComputePipelineState>,
}

impl Gpu {
    fn new() -> Self {
        let device = Device::system_default().expect("a real Metal GPU is required");
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(include_str!("../../kernels/forward_ops.metal"), &options)
            .unwrap();
        let pipelines = ["rms_kernel_cooperative", "rms_kernel_no_contract"]
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
        weights: &[f32],
        rows: usize,
        eps: f32,
        candidate: bool,
        in_place: bool,
    ) -> Vec<f32> {
        let feat = weights.len();
        assert_eq!(x.len(), rows * feat);
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = |bytes: &[f32]| {
            self.device.new_buffer_with_data(
                bytes.as_ptr().cast::<c_void>(),
                (bytes.len() * 4) as u64,
                shared,
            )
        };
        // Check writes from padded rows and element tails, in both store modes.
        let mut guarded = x.to_vec();
        guarded.extend([123456.75; 7]);
        let xb = buffer(&guarded);
        let wb = buffer(weights);
        let initial = vec![123456.75; guarded.len()];
        let yb = buffer(&initial);
        let out = if in_place { &xb } else { &yb };
        let dims = [feat as u32, rows as u32];
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines[usize::from(candidate)]);
        for (i, b) in [&xb, &wb, out].iter().enumerate() {
            encoder.set_buffer(i as u64, Some(b), 0);
        }
        encoder.set_bytes(3, 8, dims.as_ptr().cast());
        encoder.set_bytes(4, 4, (&eps as *const f32).cast());
        let groups = rows + 1;
        encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(64, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), guarded.len()) }.to_vec()
    }
}

fn fixture(feat: usize, rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut random = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / 16777216.0) * 2.0 - 1.0
    };
    let x = (0..feat * rows)
        .map(|i| {
            if seed == 0 {
                return if i % 2 == 0 { 0.0 } else { -0.0 };
            }
            let r = random();
            match i % 13 {
                0 => 0.0,
                1 => -0.0,
                2 => r * 1e4,
                3 => r * 1e-4,
                _ => r,
            }
        })
        .collect();
    let weights = (0..feat)
        .map(|i| {
            let r = random();
            match i % 9 {
                0 => -0.0,
                1 => 0.0,
                2 => r * 100.0,
                _ => r,
            }
        })
        .collect();
    (x, weights)
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn rms_no_contract_matches_cpu_at_lane_and_store_tails() {
    let gpu = Gpu::new();
    let shapes = [
        (1, 1),
        (7, 3),
        (8, 9),
        (9, 1),
        (31, 3),
        (32, 9),
        (33, 1),
        (63, 3),
        (64, 9),
        (65, 1),
        (255, 3),
        (256, 9),
        (257, 1),
        (1024, 3),
        (2048, 1),
        (4096, 3),
        (8191, 1),
        (8192, 3),
    ];
    let mut checked = 0usize;
    for (feat, rows) in shapes {
        for seed in [0, 7, 0x123456789abcdef0] {
            let (x, weights) = fixture(feat, rows, seed);
            let input = Tensor::from_f32(x.clone(), Shape::new(&[rows, feat]).unwrap()).unwrap();
            let weight = Tensor::from_f32(weights.clone(), Shape::new(&[feat]).unwrap()).unwrap();
            for eps in [1e-5f32, 1e-6f32, 0.125f32] {
                let output = input.rms_norm(&weight, eps).unwrap();
                let Storage::Cpu(storage) = output.storage() else {
                    panic!("CPU output expected");
                };
                let mut reference = storage.as_f32_slice().to_vec();
                reference.extend([123456.75; 7]);
                for candidate in [false, true] {
                    for in_place in [false, true] {
                        let actual = gpu.run(&x, &weights, rows, eps, candidate, in_place);
                        for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
                            assert_eq!(a.to_bits(), b.to_bits(),
                                "feat={feat} rows={rows} seed={seed} eps={eps} candidate={candidate} in_place={in_place} i={i}: {a} vs {b}");
                        }
                        checked += rows * feat;
                    }
                }
            }
        }
    }
    eprintln!("checked {checked} RMS scalar output bits against CPU, plus output guards");
}
