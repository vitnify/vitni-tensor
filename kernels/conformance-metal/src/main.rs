//! Deterministic-GPU conformance harness.
//!
//! Proves that the Metal kernel in `kernels/canonical_matmul.metal` produces
//! f32 output **bit-for-bit identical** to the vitni-tensor CPU canonical
//! reduction, which is the certificate's definition of "the same computation".
//!
//! Structure of the proof:
//!   1. `cpu_matmul` re-implements the contract (fixed_tree / canonical_chunk /
//!      canonical_dot) verbatim from `vitni_tensor::ops::quant`.
//!   2. It is anchored to ground truth: on the exact (4,64,4) LCG vector from
//!      `matmul.rs::matmul_reduction_bits_are_pinned`, the FNV-1a hash of the
//!      CPU output MUST equal PINNED_MATMUL_HASH. If it does, this replica IS
//!      the contract, byte-for-byte.
//!   3. The Metal kernel (compiled with fastMathEnabled = false) is run on the
//!      same inputs and compared bit-for-bit, and its output is independently
//!      re-hashed and checked against the same pin.

use std::ffi::c_void;
use metal::{
    CompileOptions, Device, MTLResourceOptions, MTLSize,
};

// ---- contract constants (must match vitni_tensor::ops::quant) ----
const CANON_LANES: usize = 8;
const CANON_CHUNK: usize = 8192;

/// The pin from `matmul.rs::matmul_reduction_bits_are_pinned`.
const PINNED_MATMUL_HASH: u64 = 0x8a42_8433_686d_13af;

// ===================== CPU reference (the contract) =====================

/// Fixed pairwise tree over `part`, in place. Verbatim shape of the CPU
/// `fixed_tree`.
fn fixed_tree(part: &mut [f32]) -> f32 {
    let mut len = part.len();
    if len == 0 {
        return 0.0;
    }
    while len > 1 {
        let half = (len + 1) / 2;
        for t in 0..half {
            let u = 2 * t;
            part[t] = if u + 1 < len { part[u] + part[u + 1] } else { part[u] };
        }
        len = half;
    }
    part[0]
}

/// canonical_chunk: 8 independent lane accumulators, element -> lane by
/// `i % CANON_LANES`, separate multiply then add (no FMA), then a fixed tree.
fn canonical_chunk(x: &[f32], w: &[f32]) -> f32 {
    let n = x.len();
    let mut lanes = [0.0f32; CANON_LANES];
    let full = n - (n % CANON_LANES);

    let mut i = 0;
    while i < full {
        for j in 0..CANON_LANES {
            let p = x[i + j] * w[i + j];
            lanes[j] += p;
        }
        i += CANON_LANES;
    }
    while i < n {
        let j = i % CANON_LANES;
        let p = x[i] * w[i];
        lanes[j] += p;
        i += 1;
    }
    fixed_tree(&mut lanes)
}

/// canonical_dot: single chunk when `n <= CANON_CHUNK`, else per-chunk sums
/// combined through a fixed tree in index order.
fn canonical_dot(x: &[f32], w: &[f32], n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    if n <= CANON_CHUNK {
        return canonical_chunk(&x[..n], &w[..n]);
    }
    let nchunks = (n + CANON_CHUNK - 1) / CANON_CHUNK;
    let mut sums = vec![0.0f32; nchunks];
    for c in 0..nchunks {
        let s = c * CANON_CHUNK;
        let e = std::cmp::min(s + CANON_CHUNK, n);
        sums[c] = canonical_chunk(&x[s..e], &w[s..e]);
    }
    fixed_tree(&mut sums)
}

/// Row-major `a[M,K] @ b[K,N]` via the canonical dot, exactly as
/// `matmul.rs::matmul` does it (gather the column, then canonical_dot).
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let mut col = vec![0.0f32; k];
    for j in 0..n {
        if k == 0 {
            continue;
        }
        if n == 1 {
            col.copy_from_slice(&b[..k]);
        } else {
            for kk in 0..k {
                col[kk] = b[kk * n + j];
            }
        }
        for i in 0..m {
            let row = &a[i * k..i * k + k];
            out[i * n + j] = canonical_dot(row, &col, k);
        }
    }
    out
}

// ===================== helpers =====================

/// FNV-1a over the raw f32 bit patterns — the exact hash from the pin test.
fn fnv1a_f32(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v.iter() {
        for byte in x.to_bits().to_le_bytes().iter() {
            h ^= *byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The exact LCG vector generator from the pin test.
fn pin_vector(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s: u64 = 0x1234;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
    (a, b)
}

/// Deterministic per-shape random inputs (different stream than the pin).
fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Monotonic total order on f32 bits, for ULP distance.
fn ord(x: f32) -> i64 {
    let b = x.to_bits();
    (if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }) as i64
}
fn ulp_diff(a: f32, b: f32) -> i64 {
    (ord(a) - ord(b)).abs()
}

// ===================== Metal =====================

struct Gpu {
    device: Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

impl Gpu {
    fn new(kernel_src: &str, fast_math: bool) -> Gpu {
        let device = Device::system_default().expect("no Metal device");
        let opts = CompileOptions::new();
        // THE knob: fast_math = false forbids fma contraction / reassociation
        // and preserves denormals, so the GPU rounds exactly like the CPU.
        opts.set_fast_math_enabled(fast_math);
        let lib = device
            .new_library_with_source(kernel_src, &opts)
            .expect("compile kernel");
        let func = lib.get_function("canonical_matmul", None).expect("get fn");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");
        let queue = device.new_command_queue();
        Gpu { device, pipeline, queue }
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let buf_a = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (a.len() * 4) as u64,
            shared,
        );
        let buf_b = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (b.len() * 4) as u64,
            shared,
        );
        let out_len = m * n;
        let buf_out = self.device.new_buffer((out_len * 4) as u64, shared);
        let dims: [u32; 3] = [m as u32, k as u32, n as u32];

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 12, dims.as_ptr() as *const c_void);

        let total = out_len as u64;
        let tg = 64u64.min(total.max(1));
        let groups = (total + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = buf_out.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec()
    }
}

// ===================== main =====================

fn main() {
    let kernel_src = include_str!("../../canonical_matmul.metal");
    let mut failures = 0;

    // ---- Step 1: anchor the CPU replica to the shipped pin ----
    let (pa, pb) = pin_vector(4, 64, 4);
    let cpu_pin = cpu_matmul(&pa, &pb, 4, 64, 4);
    let cpu_pin_hash = fnv1a_f32(&cpu_pin);
    println!("== ground truth ==");
    println!(
        "  CPU replica FNV over (4,64,4) pin vector: {:#018x}",
        cpu_pin_hash
    );
    println!("  PINNED_MATMUL_HASH (from matmul.rs):     {:#018x}", PINNED_MATMUL_HASH);
    if cpu_pin_hash == PINNED_MATMUL_HASH {
        println!("  OK: CPU replica reproduces the contract byte-for-byte.\n");
    } else {
        println!("  FAIL: CPU replica does NOT match the contract — replica is wrong.\n");
        failures += 1;
    }

    let device = Device::system_default().expect("no Metal device");
    println!("== device ==");
    println!("  {}\n", device.name());

    // ---- Step 2: GPU with fast-math OFF must match the CPU bit-for-bit ----
    let gpu = Gpu::new(kernel_src, false);

    // 2a: the pin vector, straight through the GPU.
    let gpu_pin = gpu.matmul(&pa, &pb, 4, 64, 4);
    let gpu_pin_hash = fnv1a_f32(&gpu_pin);
    let pin_bits_match = cpu_pin
        .iter()
        .zip(&gpu_pin)
        .all(|(c, g)| c.to_bits() == g.to_bits());
    println!("== Metal, fastMathEnabled = false ==");
    println!(
        "  GPU FNV over (4,64,4) pin vector:        {:#018x}  ({})",
        gpu_pin_hash,
        if gpu_pin_hash == PINNED_MATMUL_HASH { "== pin OK" } else { "!= pin FAIL" }
    );
    println!(
        "  GPU vs CPU bit-for-bit on pin vector:    {}",
        if pin_bits_match { "IDENTICAL" } else { "DIFFERS" }
    );
    if gpu_pin_hash != PINNED_MATMUL_HASH || !pin_bits_match {
        failures += 1;
    }
    println!();

    // 2b: a spread of shapes on random inputs, incl. K > CANON_CHUNK (chunked).
    let shapes: &[(usize, usize, usize, &str)] = &[
        (2, 3, 2, "tiny"),
        (4, 64, 4, "pin shape"),
        (1, 4096, 1, "single dot, mid-K"),
        (8, 512, 8, "square-ish"),
        (16, 1024, 16, "1K reduction"),
        (2, 8192, 2, "K == CANON_CHUNK (1 chunk exactly)"),
        (2, 8193, 2, "K = CANON_CHUNK+1 (spills to 2 chunks)"),
        (4, 14336, 4, "Mistral FFN K (2 chunks)"),
        (7, 333, 5, "non-multiples everywhere (tail path)"),
    ];
    println!("== bit-for-bit sweep (Metal fast-math OFF vs CPU) ==");
    println!("  {:<38} {:>10} {:>12} {:>10}", "shape", "elems", "exact-bit", "max ULP");
    for &(m, k, n, label) in shapes {
        let a = rand_vec(m * k, 0xABCD_0000 ^ (k as u64) << 8 ^ m as u64);
        let b = rand_vec(k * n, 0x1234_5678 ^ (n as u64) << 8 ^ k as u64);
        let cpu = cpu_matmul(&a, &b, m, k, n);
        let gpu = gpu.matmul(&a, &b, m, k, n);
        let exact = cpu.iter().zip(&gpu).filter(|(c, g)| c.to_bits() == g.to_bits()).count();
        let max_ulp = cpu.iter().zip(&gpu).map(|(c, g)| ulp_diff(*c, *g)).max().unwrap_or(0);
        let ok = exact == cpu.len();
        println!(
            "  {:<38} {:>10} {:>12} {:>10}  {}",
            format!("[{}x{}x{}] {}", m, k, n, label),
            cpu.len(),
            format!("{}/{}", exact, cpu.len()),
            max_ulp,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            failures += 1;
        }
    }
    println!();

    // ---- Step 3: control — show that fast-math ON is NOT safe ----
    // (Documents WHY the knob matters: the same kernel, fused, can move bits.)
    let gpu_fast = Gpu::new(kernel_src, true);
    let (a, b) = (rand_vec(4 * 14336, 0xFEED), rand_vec(14336 * 4, 0xBEEF));
    let cpu = cpu_matmul(&a, &b, 4, 14336, 4);
    let gfast = gpu_fast.matmul(&a, &b, 4, 14336, 4);
    let fast_exact = cpu.iter().zip(&gfast).filter(|(c, g)| c.to_bits() == g.to_bits()).count();
    let fast_ulp = cpu.iter().zip(&gfast).map(|(c, g)| ulp_diff(*c, *g)).max().unwrap_or(0);
    println!("== control: Metal fast-math ON (fma allowed), [4x14336x4] ==");
    println!(
        "  exact-bit {}/{}, max ULP {}  ({})",
        fast_exact,
        cpu.len(),
        fast_ulp,
        if fast_exact == cpu.len() {
            "matched anyway on this input"
        } else {
            "DIVERGES — this is the gap fast-math-off closes"
        }
    );
    println!();

    // ---- verdict ----
    if failures == 0 {
        println!("VERDICT: PASS — deterministic GPU matmul is bit-for-bit identical to the CPU contract.");
        std::process::exit(0);
    } else {
        println!("VERDICT: FAIL — {} check(s) diverged.", failures);
        std::process::exit(1);
    }
}
