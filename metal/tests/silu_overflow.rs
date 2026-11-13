//! Exact tests of the active shader against the actual CPU SiLU implementation.
use metal::{Buffer, CompileOptions, Device, MTLResourceOptions, MTLSize};
use vitni_tensor::{Shape, Storage, Tensor};

const SOURCE: &str = include_str!("../../kernels/forward_ops.metal");
const PROBE: &str = r#"
kernel void division_probe(device const float* a [[buffer(0)]],
 device const float* b [[buffer(1)]],device float* out [[buffer(2)]],
 constant uint& n [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<n) out[i]=div_sw(a[i],b[i]);
}
"#;
const PRODUCT_PROBE: &str = r#"
kernel void product_probe(device const float* a [[buffer(0)]],
 device const float* b [[buffer(1)]],device float* out [[buffer(2)]],
 constant uint& n [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<n) out[i]=vt_silu_product(a[i],b[i]);
}
"#;
struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    lib: metal::Library,
}
impl Gpu {
    fn new(source: &str) -> Self {
        let device = Device::system_default().expect("real Metal hardware is required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let product = if source.contains("vt_silu_product") {
            PRODUCT_PROBE
        } else {
            ""
        };
        let lib = device
            .new_library_with_source(&format!("{source}\n{PROBE}\n{product}"), &opts)
            .unwrap();
        let queue = device.new_command_queue();
        Self { device, queue, lib }
    }
    fn buffer(&self, values: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            values.as_ptr().cast(),
            (values.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    fn run(&self, name: &str, a: &[f32], b: Option<&[f32]>) -> Vec<f32> {
        let n = a.len();
        assert!(n > 0);
        if let Some(b) = b {
            assert_eq!(n, b.len());
        }
        let a = self.buffer(a);
        let b = b.map(|b| self.buffer(b));
        let mut initial = vec![f32::from_bits(0x4a123456); n + 3];
        let out = self.buffer(&initial);
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&self.lib.get_function(name, None).unwrap())
            .unwrap();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a), 0);
        let (output_index, n_index) = if let Some(b) = &b {
            encoder.set_buffer(1, Some(b), 0);
            (2, 3)
        } else {
            (1, 2)
        };
        encoder.set_buffer(output_index, Some(&out), 0);
        let extent = n as u32;
        encoder.set_bytes(n_index, 4, (&extent as *const u32).cast());
        encoder.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(64) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        let values = unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), n + 3) };
        for (i, v) in values[n..].iter().enumerate() {
            assert_eq!(v.to_bits(), initial[n + i].to_bits(), "output guard");
        }
        initial[..n].copy_from_slice(&values[..n]);
        initial.truncate(n);
        initial
    }
}
fn cpu_silu(x: &[f32]) -> Vec<f32> {
    let t = Tensor::from_f32(x.to_vec(), Shape::new(&[x.len()]).unwrap())
        .unwrap()
        .silu()
        .unwrap();
    let Storage::Cpu(v) = t.storage() else {
        panic!("CPU result required")
    };
    v.as_f32_slice().to_vec()
}
fn check(a: &[f32], b: &[f32], x: &[f32], label: &str) {
    let mut mismatches = 0;
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        if a.to_bits() != b.to_bits() {
            if mismatches < 8 {
                eprintln!(
                    "{label} index={i} x={:08x} actual={:08x} expected={:08x}",
                    x[i].to_bits(),
                    a.to_bits(),
                    b.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{label}: exact CPU comparison");
}

#[test]
#[ignore = "requires actual Metal GPU"]
fn finite_over_infinity_is_signed_zero() {
    metal::objc::rc::autoreleasepool(|| {
        let gpu = Gpu::new(SOURCE);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for word in [
            0, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x42b17218, 0x7f7fffff,
        ] {
            for sign in [0, 0x80000000] {
                for divisor in [f32::INFINITY, f32::NEG_INFINITY] {
                    a.push(f32::from_bits(word | sign));
                    b.push(divisor);
                }
            }
        }
        let expected: Vec<_> = a.iter().zip(&b).map(|(a, b)| a / b).collect();
        check(
            &gpu.run("division_probe", &a, Some(&b)),
            &expected,
            &a,
            "finite/inf",
        );
        eprintln!(
            "verified {} finite/inf words including both zero signs",
            a.len()
        );
    });
}

#[test]
#[ignore = "requires actual Metal GPU"]
fn finite_product_underflow_rounding_matches_cpu() {
    metal::objc::rc::autoreleasepool(|| {
        let gpu = Gpu::new(SOURCE);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for wa in [
            0, 1, 2, 3, 0x007fffff, 0x00800000, 0x00800001, 0x00ffffff, 0x01000000, 0x3f000000,
            0x3f800000, 0x7f7fffff,
        ] {
            for wb in [
                0, 1, 3, 0x007fffff, 0x00800000, 0x3effffff, 0x3f000000, 0x3f000001, 0x3f800000,
                0x7f7fffff,
            ] {
                for sign in [0, 0x80000000] {
                    a.push(f32::from_bits(wa | sign));
                    b.push(f32::from_bits(wb));
                }
            }
        }
        let mut state = 0x390173a5u32;
        for _ in 0..100000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let wa = state;
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let wb = state;
            if wa & 0x7fffffff < 0x7f800000 && wb & 0x7fffffff < 0x7f800000 {
                a.push(f32::from_bits(wa));
                b.push(f32::from_bits(wb));
            }
        }
        let expected: Vec<_> = a.iter().zip(&b).map(|(a, b)| a * b).collect();
        check(
            &gpu.run("product_probe", &a, Some(&b)),
            &expected,
            &a,
            "finite product",
        );
        eprintln!(
            "verified {} finite product words including rounding boundaries",
            a.len()
        );
    });
}

#[test]
#[ignore = "requires actual Metal GPU"]
fn silu_overflow_boundary_and_finite_ranges_match_cpu() {
    metal::objc::rc::autoreleasepool(|| {
        let gpu = Gpu::new(SOURCE);
        let mut gates = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            -87.,
            -88.,
            -89.,
            -90.,
            -100.,
            -1000.,
            -1e10,
            -1e30,
            f32::MAX,
            -f32::MAX,
        ];
        for word in [1, 2, 3, 0x007fffff, 0x00800000, 0x00800001, 0x01000000] {
            gates.push(f32::from_bits(word));
            gates.push(f32::from_bits(word | 0x80000000));
        }
        for offset in -8192i64..=8192 {
            gates.push(f32::from_bits((0xc2b17218i64 + offset) as u32));
        }
        let mut state = 0x18394275u32;
        for _ in 0..32768 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            if state & 0x7fffffff < 0x7f800000 {
                gates.push(f32::from_bits(state));
            }
        }
        let reference = cpu_silu(&gates);
        check(
            &gpu.run("silu_kernel", &gates, None),
            &reference,
            &gates,
            "SiLU",
        );
        let multipliers = [
            0.,
            -0.,
            1.,
            -1.,
            0.5,
            -0.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-20,
            -1e20,
            f32::MAX,
        ];
        let mut fused_words = 0;
        for offset in 0..multipliers.len() {
            let up: Vec<_> = (0..gates.len())
                .map(|i| multipliers[(i + offset) % multipliers.len()])
                .collect();
            let expected: Vec<_> = reference.iter().zip(&up).map(|(s, u)| s * u).collect();
            check(
                &gpu.run("silu_mul", &gates, Some(&up)),
                &expected,
                &gates,
                "SwiGLU",
            );
            fused_words += gates.len();
        }
        eprintln!(
            "verified {} SiLU and {fused_words} SwiGLU words, plus output guards",
            gates.len()
        );
    });
}

#[test]
#[ignore = "requires actual Metal GPU"]
fn frozen_original_shader_demonstrates_the_regression() {
    metal::objc::rc::autoreleasepool(|| {
        let Some(path) = std::env::var_os("VITNI_SILU_BASELINE_SOURCE") else {
            eprintln!("baseline contrast not requested; fixed-shader tests remain mandatory");
            return;
        };
        let baseline = Gpu::new(&std::fs::read_to_string(path).unwrap());
        let fixed = Gpu::new(SOURCE);
        let gates = [-89., -90., -100., -1000., -1e10, -f32::MAX];
        let reference = cpu_silu(&gates);
        let old = baseline.run("silu_kernel", &gates, None);
        let mismatches = old
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(mismatches > 0, "old shader must actually reproduce the bug");
        check(
            &fixed.run("silu_kernel", &gates, None),
            &reference,
            &gates,
            "fixed overflow",
        );
        eprintln!(
            "old shader had {mismatches}/{} mismatches; fixed shader has zero",
            gates.len()
        );
    });
}
