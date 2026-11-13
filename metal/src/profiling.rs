//! Opt-in GPU timing. Dispatch mode deliberately splits compute passes; compare
//! its overhead against command mode before interpreting the attribution.
use metal::{
    objc::{msg_send, runtime::Object, sel, sel_impl},
    CommandBufferRef, ComputePassDescriptor, ComputePipelineState, CounterSampleBuffer,
    CounterSampleBufferDescriptor, DeviceRef, MTLCounterSamplingPoint, MTLStorageMode, NSRange,
};
use std::{cell::RefCell, time::Instant};

#[derive(Clone, Debug)]
pub struct DispatchTiming {
    pub kernel: String,
    pub threads: usize,
    pub dimensions: Vec<Vec<u32>>,
    pub gpu_seconds: f64,
    pub gpu_start_tick: u64,
    pub gpu_end_tick: u64,
}

#[derive(Clone, Debug)]
pub struct StepTiming {
    pub mode: &'static str,
    pub prepare_seconds: f64,
    pub encode_seconds: f64,
    pub commit_wait_seconds: f64,
    pub readback_seconds: f64,
    pub gpu_seconds: f64,
    pub resolve_seconds: f64,
    pub gpu_tick_nanoseconds: f64,
    pub dispatches: Vec<DispatchTiming>,
}

pub(crate) struct Profiler {
    pub split: bool,
    samples: Option<CounterSampleBuffer>,
    records: RefCell<Vec<DispatchTiming>>,
    last: RefCell<Option<StepTiming>>,
}

impl Profiler {
    pub fn from_environment(device: &DeviceRef) -> Result<Option<Self>, String> {
        let mode = std::env::var("VITNI_METAL_PROFILE").unwrap_or_default();
        if mode.is_empty() || mode == "off" {
            return Ok(None);
        }
        if mode != "command" && mode != "dispatch" {
            return Err("VITNI_METAL_PROFILE must be off, command or dispatch".into());
        }
        let split = mode == "dispatch";
        let samples = if split {
            if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
                return Err("GPU lacks stage-boundary timestamp sampling".into());
            }
            let set = device
                .counter_sets()
                .into_iter()
                .find(|s| s.name() == "timestamp")
                .ok_or("GPU lacks the Timestamp counter set")?;
            let descriptor = CounterSampleBufferDescriptor::new();
            descriptor.set_counter_set(&set);
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            // 2048 eight-byte timestamps, below Apple's 32 KiB limit. Fail on
            // larger graphs instead of silently omitting any operation.
            descriptor.set_sample_count(2048);
            Some(device.new_counter_sample_buffer_with_descriptor(&descriptor)?)
        } else {
            None
        };
        Ok(Some(Self {
            split,
            samples,
            records: RefCell::new(Vec::new()),
            last: RefCell::new(None),
        }))
    }

    pub fn begin(&self) {
        self.records.borrow_mut().clear();
    }

    pub fn encode_dispatch(
        &self,
        cmd: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        scalars: &[(u64, &[u8])],
        threads: usize,
        encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
    ) {
        let mut records = self.records.borrow_mut();
        let index = records.len() * 2;
        assert!(index + 1 < 2048, "profiling counter capacity exceeded");
        let descriptor = ComputePassDescriptor::new();
        let attachment = descriptor.sample_buffer_attachments().object_at(0).unwrap();
        attachment.set_sample_buffer(self.samples.as_ref().unwrap());
        attachment.set_start_of_encoder_sample_index(index as u64);
        attachment.set_end_of_encoder_sample_index(index as u64 + 1);
        let encoder = cmd.compute_command_encoder_with_descriptor(descriptor);
        encode(encoder);
        encoder.end_encoding();
        records.push(DispatchTiming {
            kernel: pipeline.label().to_owned(),
            threads,
            dimensions: scalars
                .iter()
                .map(|(_, bytes)| {
                    bytes
                        .chunks_exact(4)
                        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                        .collect()
                })
                .collect(),
            gpu_seconds: 0.0,
            gpu_start_tick: 0,
            gpu_end_tick: 0,
        });
    }

    pub fn finish(
        &self,
        device: &DeviceRef,
        cmd: &CommandBufferRef,
        start_clocks: (u64, u64),
        prepare_seconds: f64,
        encode_seconds: f64,
        commit_wait_seconds: f64,
        readback_seconds: f64,
    ) {
        let start = Instant::now();
        let mut cpu1 = 0;
        let mut gpu1 = 0;
        device.sample_timestamps(&mut cpu1, &mut gpu1);
        // sampleTimestamps CPU values are already nanoseconds, not raw Mach
        // ticks: https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time
        assert!(cpu1 > start_clocks.0 && gpu1 > start_clocks.1);
        let tick_ns = (cpu1 - start_clocks.0) as f64 / (gpu1 - start_clocks.1) as f64;
        let mut records = self.records.borrow().clone();
        if let Some(samples) = &self.samples {
            let count = records.len() * 2;
            unsafe {
                let data: *mut Object =
                    msg_send![samples.as_ref(), resolveCounterRange: NSRange::new(0, count as u64)];
                assert!(!data.is_null(), "timestamp resolution failed");
                let length: usize = msg_send![data, length];
                assert_eq!(length, count * 8);
                let bytes: *const u8 = msg_send![data, bytes];
                let values = std::slice::from_raw_parts(bytes, length);
                for (i, record) in records.iter_mut().enumerate() {
                    let a = u64::from_ne_bytes(values[i * 16..i * 16 + 8].try_into().unwrap());
                    let b = u64::from_ne_bytes(values[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                    assert!(
                        a != u64::MAX && b != u64::MAX && b >= a,
                        "invalid GPU timestamp"
                    );
                    record.gpu_seconds = (b - a) as f64 * tick_ns * 1e-9;
                    record.gpu_start_tick = a;
                    record.gpu_end_tick = b;
                }
            }
        }
        let gpu_start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let gpu_end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        assert!(
            gpu_start > 0.0 && gpu_end >= gpu_start,
            "command GPU timestamps unavailable"
        );
        *self.last.borrow_mut() = Some(StepTiming {
            mode: if self.split { "dispatch" } else { "command" },
            prepare_seconds,
            encode_seconds,
            commit_wait_seconds,
            readback_seconds,
            gpu_seconds: gpu_end - gpu_start,
            resolve_seconds: start.elapsed().as_secs_f64(),
            gpu_tick_nanoseconds: tick_ns,
            dispatches: records,
        });
    }

    pub fn take(&self) -> Option<StepTiming> {
        self.last.borrow_mut().take()
    }
}
