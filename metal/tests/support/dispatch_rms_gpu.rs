use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CompileOptions, Device, MTLCommandBufferStatus, MTLDispatchType, MTLResourceOptions,
    MTLSize,
};
use std::{ffi::c_void, time::Instant};

const Q4: &str = include_str!("../../../kernels/q4k.metal");
const FWD: &str = include_str!("../../../kernels/forward_ops.metal");
const DOT: &str = "";
const FUSION: &str = include_str!("../fixtures/rms_q4_fusion.metal");

pub fn check(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label} word={i}: {x} vs {y}");
    }
}

pub struct Gpu {
    pub device: Device,
    queue: metal::CommandQueue,
    q4: Vec<metal::ComputePipelineState>,
    rms: metal::ComputePipelineState,
    fused: Vec<metal::ComputePipelineState>,
}

impl Gpu {
    pub fn new() -> Self {
        let device = Device::system_default().expect("real Metal GPU required");
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let source = format!("#define VITNI_Q4K_NO_CONTRACT 1\n{Q4}\n{FWD}\n{DOT}\n{FUSION}");
        let lib = device
            .new_library_with_source(&source, &options)
            .expect("compile candidates");
        let pipeline = |name: &str| {
            device
                .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
                .unwrap()
        };
        let q4 = ["q4k_linear_metadata", "q4k_linear_metadata_acc"]
            .iter()
            .map(|n| pipeline(n))
            .collect::<Vec<_>>();
        let fused = ["rms_q4_fused", "rms_q4_fused_acc"]
            .iter()
            .map(|n| pipeline(n))
            .collect::<Vec<_>>();
        for p in q4.iter().chain(fused.iter()) {
            assert!(p.max_total_threads_per_threadgroup() >= 64);
        }
        let rms = pipeline("rms_kernel_no_contract");
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            q4,
            rms,
            fused,
        }
    }

    pub fn buffer(&self, bytes: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            bytes.as_ptr().cast::<c_void>(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    pub fn floats(&self, values: &[f32]) -> Buffer {
        self.buffer(
            &values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    fn read(&self, b: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents().cast::<f32>(), n) }.to_vec()
    }
    fn elapsed(&self, c: &metal::CommandBufferRef, start: Instant, reps: usize) -> (f64, f64) {
        c.commit();
        c.wait_until_completed();
        assert_eq!(c.status(), MTLCommandBufferStatus::Completed);
        let a: f64 = unsafe { msg_send![c, GPUStartTime] };
        let b: f64 = unsafe { msg_send![c, GPUEndTime] };
        assert!(a > 0.0 && b > a);
        (
            start.elapsed().as_secs_f64() / reps as f64,
            (b - a) / reps as f64,
        )
    }
    fn q4_dispatch(
        &self,
        e: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        k: usize,
        m: usize,
        acc: bool,
    ) {
        e.set_compute_pipeline_state(&self.q4[usize::from(acc)]);
        for (i, b) in [x, w, y].iter().enumerate() {
            e.set_buffer(i as u64, Some(b), 0);
        }
        let dims = [k as u32, m as u32, (k / 256) as u32];
        e.set_bytes(3, 12, dims.as_ptr().cast());
        e.dispatch_thread_groups(
            MTLSize::new((m * 8).div_ceil(64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    pub fn pair(
        &self,
        x: &Buffer,
        w: [&Buffer; 2],
        initial: &[f32],
        k: usize,
        m: usize,
        concurrent: bool,
        reps: usize,
    ) -> (Vec<f32>, Vec<f32>, f64, f64) {
        let ys = [self.floats(initial), self.floats(initial)];
        let start = Instant::now();
        let c = self.queue.new_command_buffer();
        let e = c.compute_command_encoder_with_dispatch_type(if concurrent {
            MTLDispatchType::Concurrent
        } else {
            MTLDispatchType::Serial
        });
        for _ in 0..reps {
            self.q4_dispatch(e, x, w[0], &ys[0], k, m, false);
            self.q4_dispatch(e, x, w[1], &ys[1], k, m, false);
            // Same barrier in both modes: next repetition may overwrite outputs.
            e.memory_barrier_with_resources(&[&ys[0], &ys[1]]);
        }
        e.end_encoding();
        let (h, g) = self.elapsed(c, start, reps);
        (
            self.read(&ys[0], initial.len()),
            self.read(&ys[1], initial.len()),
            h,
            g,
        )
    }

    pub fn fusion(
        &self,
        x: &Buffer,
        rw: &Buffer,
        w: &Buffer,
        initial: &[f32],
        k: usize,
        m: usize,
        eps: f32,
        fused: bool,
        acc: bool,
        reps: usize,
    ) -> (Vec<f32>, f64, f64) {
        assert!(k > 0 && k <= 4096 && k % 256 == 0);
        assert!(
            self.fused[usize::from(acc)].static_threadgroup_memory_length() + k as u64 * 4
                <= self.device.max_threadgroup_memory_length()
        );
        let y = self.floats(initial);
        let norm = self
            .device
            .new_buffer((k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let start = Instant::now();
        let c = self.queue.new_command_buffer();
        let e = c.new_compute_command_encoder();
        for _ in 0..reps {
            if fused {
                e.set_compute_pipeline_state(&self.fused[usize::from(acc)]);
                for (i, b) in [x, rw, w, &y].iter().enumerate() {
                    e.set_buffer(i as u64, Some(b), 0);
                }
                let dims = [k as u32, m as u32, (k / 256) as u32];
                e.set_bytes(4, 12, dims.as_ptr().cast());
                e.set_bytes(5, 4, (&eps as *const f32).cast());
                e.set_threadgroup_memory_length(0, (k * 4) as u64);
                e.dispatch_thread_groups(
                    MTLSize::new((m * 8).div_ceil(64) as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                e.set_compute_pipeline_state(&self.rms);
                for (i, b) in [x, rw, &norm].iter().enumerate() {
                    e.set_buffer(i as u64, Some(b), 0);
                }
                let dims = [k as u32, 1u32];
                e.set_bytes(3, 8, dims.as_ptr().cast());
                e.set_bytes(4, 4, (&eps as *const f32).cast());
                e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
                e.memory_barrier_with_resources(&[&norm]);
                self.q4_dispatch(e, &norm, w, &y, k, m, acc);
            }
            e.memory_barrier_with_resources(&[&y]);
        }
        e.end_encoding();
        let (h, g) = self.elapsed(c, start, reps);
        (self.read(&y, initial.len()), h, g)
    }
}
