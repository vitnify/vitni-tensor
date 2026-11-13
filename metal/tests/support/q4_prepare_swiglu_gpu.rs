use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, CompileOptions, Device, MTLCommandBufferStatus, MTLDispatchType, MTLResourceOptions,
    MTLSize,
};
use std::{ffi::c_void, time::Instant};

const Q4: &str = include_str!("../../../kernels/q4k.metal");
const FWD: &str = include_str!("../../../kernels/forward_ops.metal");
const PREPARE: &str = include_str!("../fixtures/q4_product_prepare.metal");
const FUSION: &str = include_str!("../fixtures/q4_gate_up_swiglu.metal");

pub fn check(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{label} word={i}: {a} vs {b}");
    }
}

pub struct Gpu {
    pub device: Device,
    queue: metal::CommandQueue,
    q4: Vec<metal::ComputePipelineState>,
    prepare: Vec<metal::ComputePipelineState>,
    silu: metal::ComputePipelineState,
    fusion: Vec<metal::ComputePipelineState>,
}

impl Gpu {
    pub fn new() -> Self {
        let device = Device::system_default().expect("real Metal GPU required; no skip");
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let prefix = "#define VITNI_Q4K_NO_CONTRACT 1\n";
        // Baselines are compiled exactly as production: Q4 and forward sources
        // separately. Candidate FWD helpers precede Q4's contraction pragma.
        let qlib = device
            .new_library_with_source(&format!("{prefix}{Q4}"), &options)
            .expect("baseline Q4");
        let flib = device
            .new_library_with_source(FWD, &options)
            .expect("baseline forward");
        let plib = device
            .new_library_with_source(&format!("{prefix}{Q4}\n{PREPARE}"), &options)
            .expect("product preparation");
        let glib = device
            .new_library_with_source(&format!("{FWD}\n{prefix}{Q4}\n{FUSION}"), &options)
            .expect("gate/up fusion");
        let pipeline = |lib: &metal::LibraryRef, name: &str| {
            let p = device
                .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
                .unwrap();
            assert!(p.max_total_threads_per_threadgroup() >= 64);
            assert_eq!(
                p.thread_execution_width(),
                32,
                "require actual SIMD width 32"
            );
            p
        };
        let q4 = ["q4k_linear_metadata", "q4k_linear_metadata_acc"]
            .iter()
            .map(|n| pipeline(&qlib, n))
            .collect();
        let prepare = [
            "q4k_linear_product_prepare",
            "q4k_linear_product_prepare_acc",
        ]
        .iter()
        .map(|n| pipeline(&plib, n))
        .collect();
        let silu = pipeline(&flib, "silu_mul");
        let fusion = ["q4_gate_up_swiglu", "q4_gate_up_swiglu_debug"]
            .iter()
            .map(|n| pipeline(&glib, n))
            .collect();
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            q4,
            prepare,
            silu,
            fusion,
        }
    }

    pub fn buffer(&self, v: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr().cast::<c_void>(),
            v.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    pub fn floats(&self, v: &[f32]) -> Buffer {
        self.buffer(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
    }
    fn read(&self, b: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents().cast::<f32>(), n) }.to_vec()
    }
    fn elapsed(&self, c: &metal::CommandBufferRef, t: Instant, reps: usize) -> (f64, f64) {
        c.commit();
        c.wait_until_completed();
        assert_eq!(c.status(), MTLCommandBufferStatus::Completed);
        let a: f64 = unsafe { msg_send![c, GPUStartTime] };
        let b: f64 = unsafe { msg_send![c, GPUEndTime] };
        assert!(a > 0.0 && b > a);
        (
            t.elapsed().as_secs_f64() / reps as f64,
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
        candidate: bool,
        acc: bool,
    ) {
        assert!(k > 0 && k % 256 == 0 && k <= 64 * 8192);
        e.set_compute_pipeline_state(if candidate {
            &self.prepare[acc as usize]
        } else {
            &self.q4[acc as usize]
        });
        for (i, b) in [x, w, y].iter().enumerate() {
            e.set_buffer(i as u64, Some(b), 0);
        }
        let dims = [k as u32, m as u32, (k / 256) as u32];
        e.set_bytes(3, 12, dims.as_ptr().cast());
        let threads = m * if candidate { 32 } else { 8 };
        e.dispatch_thread_groups(
            MTLSize::new(threads.div_ceil(64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }
    pub fn product(
        &self,
        x: &Buffer,
        w: &Buffer,
        initial: &[f32],
        k: usize,
        m: usize,
        candidate: bool,
        acc: bool,
        reps: usize,
    ) -> (Vec<f32>, f64, f64) {
        assert!(
            !acc || reps == 1,
            "residual timing would otherwise compound"
        );
        let y = self.floats(initial);
        let start = Instant::now();
        let c = self.queue.new_command_buffer();
        let e = c.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
        for _ in 0..reps {
            self.q4_dispatch(e, x, w, &y, k, m, candidate, acc);
            e.memory_barrier_with_resources(&[&y]);
        }
        e.end_encoding();
        let (h, g) = self.elapsed(c, start, reps);
        (self.read(&y, initial.len()), h, g)
    }
    pub fn fused(
        &self,
        x: &Buffer,
        w: [&Buffer; 2],
        initial: &[f32],
        k: usize,
        m: usize,
        candidate: bool,
        debug: bool,
        reps: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, f64, f64) {
        assert!(k > 0 && k % 256 == 0 && k <= 64 * 8192);
        let y = self.floats(initial);
        let gate = self.floats(initial);
        let up = self.floats(initial);
        let start = Instant::now();
        let c = self.queue.new_command_buffer();
        let e = c.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
        for _ in 0..reps {
            if candidate {
                e.set_compute_pipeline_state(&self.fusion[debug as usize]);
                for (i, b) in [x, w[0], w[1], &y].iter().enumerate() {
                    e.set_buffer(i as u64, Some(b), 0);
                }
                let dims = [k as u32, m as u32, (k / 256) as u32];
                e.set_bytes(4, 12, dims.as_ptr().cast());
                if debug {
                    e.set_buffer(5, Some(&gate), 0);
                    e.set_buffer(6, Some(&up), 0);
                }
                e.dispatch_thread_groups(
                    MTLSize::new((m * 16).div_ceil(64) as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                self.q4_dispatch(e, x, w[0], &gate, k, m, false, false);
                self.q4_dispatch(e, x, w[1], &up, k, m, false, false);
                e.memory_barrier_with_resources(&[&gate, &up]);
                e.set_compute_pipeline_state(&self.silu);
                for (i, b) in [&gate, &up, &y].iter().enumerate() {
                    e.set_buffer(i as u64, Some(b), 0);
                }
                let n = m as u32;
                e.set_bytes(3, 4, (&n as *const u32).cast());
                e.dispatch_thread_groups(
                    MTLSize::new(m.div_ceil(64) as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
            // Also protects gate/up reads before the next baseline overwrite.
            e.memory_barrier_with_resources(&[&gate, &up, &y]);
        }
        e.end_encoding();
        let (h, g) = self.elapsed(c, start, reps);
        (
            self.read(&y, initial.len()),
            self.read(&gate, initial.len()),
            self.read(&up, initial.len()),
            h,
            g,
        )
    }
}
