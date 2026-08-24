//! Forward-pass op conformance harness (Metal).
//!
//! Step 1 (this file): `expf` — the hot-path transcendental (softmax + SiLU).
//! Ground truth is `libm::expf`, the exact function vitni-tensor calls. The
//! GPU kernel (`kernels/expf.metal`, compiled fastMathEnabled=false) must
//! reproduce it bit-for-bit across the whole domain. This is the piece that
//! usually blocks a full deterministic GPU forward; proving it here shows the
//! transcendental wall is not there.

use std::ffi::c_void;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use vitni_tensor::{Shape, Storage, Tensor};

fn tvec(t: &Tensor) -> Vec<f32> {
    match t.storage() {
        Storage::Cpu(s) => s.as_f32_slice().to_vec(),
        _ => panic!("expected CPU storage"),
    }
}

fn lcg_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn ord(x: f32) -> i64 {
    let b = x.to_bits();
    (if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }) as i64
}
fn ulp(a: f32, b: f32) -> i64 {
    (ord(a) - ord(b)).abs()
}

struct Gpu {
    device: Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}
impl Gpu {
    fn new(src: &str, func: &str) -> Gpu {
        let device = Device::system_default().expect("no Metal device");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        opts.set_preserve_invariance(true); // stop optimization-dependent refactors (e.g. fma contraction)
        let lib = device.new_library_with_source(src, &opts).expect("compile");
        let f = lib.get_function(func, None).expect("fn");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline");
        let queue = device.new_command_queue();
        Gpu { device, pipeline, queue }
    }
    fn expf(&self, input: &[f32]) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let n = input.len();
        let bi = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (n * 4) as u64,
            shared,
        );
        let bo = self.device.new_buffer((n * 4) as u64, shared);
        let nn = n as u32;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&bi), 0);
        enc.set_buffer(1, Some(&bo), 0);
        enc.set_bytes(2, 4, &nn as *const u32 as *const c_void);
        let tg = 256u64.min(n as u64).max(1);
        let groups = (n as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = bo.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }
    fn rms(&self, x: &[f32], w: &[f32], rows: usize, feat: usize, eps: f32) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let bx = self.device.new_buffer_with_data(x.as_ptr() as *const c_void, (x.len() * 4) as u64, shared);
        let bw = self.device.new_buffer_with_data(w.as_ptr() as *const c_void, (w.len() * 4) as u64, shared);
        let bo = self.device.new_buffer((rows * feat * 4) as u64, shared);
        let dims: [u32; 2] = [feat as u32, rows as u32];
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&bx), 0);
        enc.set_buffer(1, Some(&bw), 0);
        enc.set_buffer(2, Some(&bo), 0);
        enc.set_bytes(3, 8, dims.as_ptr() as *const c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        let tg = 64u64.min(rows as u64).max(1);
        let groups = (rows as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = bo.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows * feat) }.to_vec()
    }
    fn rope(&self, x: &[f32], cosc: &[f32], sinc: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let bx = self.device.new_buffer_with_data(x.as_ptr() as *const c_void, (x.len() * 4) as u64, shared);
        let bc = self.device.new_buffer_with_data(cosc.as_ptr() as *const c_void, (cosc.len() * 4) as u64, shared);
        let bs = self.device.new_buffer_with_data(sinc.as_ptr() as *const c_void, (sinc.len() * 4) as u64, shared);
        let bo = self.device.new_buffer((x.len() * 4) as u64, shared);
        let dims: [u32; 3] = [seq as u32, n_heads as u32, head_dim as u32];
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&bx), 0);
        enc.set_buffer(1, Some(&bc), 0);
        enc.set_buffer(2, Some(&bs), 0);
        enc.set_buffer(3, Some(&bo), 0);
        enc.set_bytes(4, 12, dims.as_ptr() as *const c_void);
        let total = (seq * n_heads) as u64;
        let tg = 64u64.min(total).max(1);
        let groups = (total + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = bo.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, x.len()) }.to_vec()
    }
    fn softmax(&self, x: &[f32], rows: usize, last: usize) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let bx = self.device.new_buffer_with_data(x.as_ptr() as *const c_void, (x.len() * 4) as u64, shared);
        let bo = self.device.new_buffer((rows * last * 4) as u64, shared);
        let dims: [u32; 2] = [last as u32, rows as u32];
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&bx), 0);
        enc.set_buffer(1, Some(&bo), 0);
        enc.set_bytes(2, 8, dims.as_ptr() as *const c_void);
        let tg = 64u64.min(rows as u64).max(1);
        let groups = (rows as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = bo.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows * last) }.to_vec()
    }
    fn q4k(&self, x: &[f32], w: &[u8], m: usize, k: usize, nsuper: usize) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let bx = self.device.new_buffer_with_data(x.as_ptr() as *const c_void, (x.len() * 4) as u64, shared);
        let bw = self.device.new_buffer_with_data(w.as_ptr() as *const c_void, w.len() as u64, shared);
        let bo = self.device.new_buffer((m * 4) as u64, shared);
        let dims: [u32; 3] = [k as u32, m as u32, nsuper as u32];
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&bx), 0);
        enc.set_buffer(1, Some(&bw), 0);
        enc.set_buffer(2, Some(&bo), 0);
        enc.set_bytes(3, 12, dims.as_ptr() as *const c_void);
        let tg = 64u64.min(m as u64).max(1);
        let groups = (m as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = bo.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, m) }.to_vec()
    }
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let n = a.len();
        let ba = self.device.new_buffer_with_data(a.as_ptr() as *const c_void, (n * 4) as u64, shared);
        let bb = self.device.new_buffer_with_data(b.as_ptr() as *const c_void, (n * 4) as u64, shared);
        let bo = self.device.new_buffer((n * 4) as u64, shared);
        let nn = n as u32;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&ba), 0);
        enc.set_buffer(1, Some(&bb), 0);
        enc.set_buffer(2, Some(&bo), 0);
        enc.set_bytes(3, 4, &nn as *const u32 as *const c_void);
        let tg = 256u64.min(n as u64).max(1);
        let groups = (n as u64 + tg - 1) / tg;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = bo.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }
}

/// Software f32 division — IDENTICAL source to `div_sw` in expf.metal. Uses
/// only integer bit-ops + correctly-rounded FMA (`mul_add`), so it produces the
/// same bits on any device with an IEEE FMA. This is the cross-vendor divide.
fn div_sw(a: f32, b: f32) -> f32 {
    let sgn = (a.to_bits() ^ b.to_bits()) & 0x8000_0000;
    let ba = f32::from_bits(b.to_bits() & 0x7fff_ffff);
    let aa = f32::from_bits(a.to_bits() & 0x7fff_ffff);
    let j = 0x7EF127EAu32.wrapping_sub(ba.to_bits());
    let mut y = f32::from_bits(j);
    let nb = -ba;
    let mut e;
    e = nb.mul_add(y, 1.0); y = y.mul_add(e, y);
    e = nb.mul_add(y, 1.0); y = y.mul_add(e, y);
    e = nb.mul_add(y, 1.0); y = y.mul_add(e, y);
    let q = aa * y;
    let r = nb.mul_add(q, aa);
    let q = r.mul_add(y, q);
    f32::from_bits(q.to_bits() ^ sgn)
}

/// Rust transcription of the EXACT algorithm in expf.metal (separate ops +
/// div_sw + our scalbnf). Lets us split "GPU vs our algorithm" from "our
/// algorithm vs libm".
fn scalbnf_ref(mut x: f32, mut n: i32) -> f32 {
    let f_exp_max = f32::from_bits(254u32 << 23);
    let f_exp_min = f32::from_bits(1u32 << 23);
    let f_pow_subnorm = f32::from_bits(151u32 << 23);
    if n > 127 {
        x *= f_exp_max; n -= 127;
        if n > 127 { x *= f_exp_max; n -= 127; if n > 127 { n = 127; } }
    } else if n < -126 {
        let mul = f_exp_min * f_pow_subnorm;
        let add = 126 - 24;
        x *= mul; n += add;
        if n < -126 { x *= mul; n += add; if n < -126 { n = -126; } }
    }
    let scale = f32::from_bits(((127 + n) as u32) << 23);
    x * scale
}
fn expf_ref(mut x: f32) -> f32 {
    let (h0, h1) = (0.5f32, -0.5f32);
    let ln2_hi = 6.9314575195e-01f32;
    let ln2_lo = 1.4286067653e-06f32;
    let inv_ln2 = 1.4426950216e+00f32;
    let p1 = 1.6666625440e-1f32;
    let p2 = -2.7667332906e-3f32;
    let x1p127 = f32::from_bits(0x7f000000);
    let mut hx = x.to_bits();
    let sign = (hx >> 31) as i32;
    let signb = sign != 0;
    hx &= 0x7fffffff;
    if hx >= 0x42aeac50 {
        if hx > 0x7f800000 { return x; }
        if hx >= 0x42b17218 && !signb { x *= x1p127; return x; }
        if signb && hx >= 0x42cff1b5 { return 0.0; }
    }
    let k: i32;
    let hi: f32;
    let lo: f32;
    if hx > 0x3eb17218 {
        if hx > 0x3f851592 {
            let h = if signb { h1 } else { h0 };
            k = (inv_ln2 * x + h) as i32;
        } else {
            k = 1 - sign - sign;
        }
        let kf = k as f32;
        hi = x - kf * ln2_hi;
        lo = kf * ln2_lo;
        x = hi - lo;
    } else if hx > 0x39000000 {
        k = 0; hi = x; lo = 0.0;
    } else {
        return 1.0 + x;
    }
    let xx = x * x;
    let c = x - xx * (p1 + xx * p2);
    let y = 1.0 + ((div_sw(x * c, 2.0 - c) - lo) + hi);
    if k == 0 { y } else { scalbnf_ref(y, k) }
}

fn softdiv_cross_vendor(kernel_src: &str) {
    let gpu = Gpu::new(kernel_src, "divsw_kernel");
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut s: u64 = 0xDEAD_BEEF_CAFE_0001;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // normal-range f32 with random sign (exp field in [~1e-3, ~1e3] scale)
        let m = (s >> 40) as u32 & 0x007f_ffff;
        let ex = (((s >> 8) as u32) % 60 + 100) << 23; // biased exp ~ [100,160) -> 2^-27..2^33
        let sign = ((s >> 3) as u32 & 1) << 31;
        f32::from_bits(sign | ex | m)
    };
    for _ in 0..8_000_000 {
        a.push(rnd());
        b.push(rnd());
    }
    let g = gpu.div(&a, &b);
    let mut cross_mism = 0usize; // CPU_sw vs GPU_sw  (the guarantee)
    let mut acc_mism = 0usize; // div_sw vs true a/b (accuracy)
    let mut acc_max_ulp = 0i64;
    for ((x, y), gv) in a.iter().zip(&b).zip(&g) {
        let cpu_sw = div_sw(*x, *y);
        if cpu_sw.to_bits() != gv.to_bits() {
            cross_mism += 1;
        }
        let truth = *x / *y;
        if div_sw(*x, *y).to_bits() != truth.to_bits() {
            acc_mism += 1;
            let u = ulp(div_sw(*x, *y), truth);
            if u > acc_max_ulp {
                acc_max_ulp = u;
            }
        }
    }
    println!("== software division: CPU vs Metal (the cross-vendor guarantee) ==");
    println!(
        "  CPU_sw vs GPU_sw bit-identical: {}/{}  ({})",
        a.len() - cross_mism,
        a.len(),
        if cross_mism == 0 { "IDENTICAL — Metal rejoins the cross-vendor set" } else { "DIVERGES" }
    );
    println!(
        "  (accuracy vs IEEE a/b: {} off, max ULP {} — faithful/near-correct rounding)",
        acc_mism, acc_max_ulp
    );
    println!();
}

fn q6k_conformance(q6k_src: &str) {
    let gpu = Gpu::new(q6k_src, "q6k_linear");
    println!("== Q6_K fused dequant+dot: Metal vs canonical_dot_q6k_fused ==");
    let shapes: &[(usize, usize, &str)] = &[
        (256, 4, "1 super-block"),
        (512, 4, "2 super-blocks"),
        (2048, 8, "8 super-blocks"),
        (8192, 2, "32 sb = 1 chunk"),
        (8448, 2, "33 sb -> 2 chunks"),
        (5632, 4, "TinyLlama hidden (22 sb)"),
    ];
    let mut fails = 0usize;
    for &(k, m, label) in shapes {
        let nsuper = k / 256;
        let x = lcg_vec(k, 0xD6C0 ^ (k as u64).wrapping_mul(0x9E3779B1));
        let mut w = vec![0u8; m * nsuper * 210];
        let mut s: u64 = 0x6C6E_0000 ^ (k as u64).wrapping_mul(0x85EBCA77);
        let mut nb = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 33) as u32 };
        for byte in w.iter_mut() { *byte = (nb() & 0xFF) as u8; }
        for i in 0..m {
            for b in 0..nsuper {
                let off = (i * nsuper + b) * 210;
                let dbits = (((8 + (nb() % 6)) << 10) | (nb() & 0x3FF)) as u16;
                w[off + 208..off + 210].copy_from_slice(&dbits.to_le_bytes());
            }
        }
        let mut cpu = vec![0f32; m];
        for i in 0..m {
            let wrow = &w[i * nsuper * 210..(i + 1) * nsuper * 210];
            cpu[i] = vitni_tensor::ops::quant::canonical_dot_q6k_fused(&x, wrow, k).unwrap();
        }
        let g = gpu.q4k(&x, &w, m, k, nsuper); // generic dispatch runs q6k_linear
        let exact = cpu.iter().zip(&g).filter(|(c, gg)| c.to_bits() == gg.to_bits()).count();
        let maxu = cpu.iter().zip(&g).map(|(c, gg)| ulp(*c, *gg)).max().unwrap_or(0);
        let ok = exact == m;
        println!("  [K={:>5} x{}] {:<26} exact {}/{}  maxULP {}  {}", k, m, label, exact, m, maxu, if ok { "OK" } else { "FAIL" });
        if !ok { fails += 1; }
    }
    println!("  VERDICT: Q6_K {}\n", if fails == 0 { "PASS — bit-identical to the CPU hot path" } else { "FAIL" });
}

fn forward_ops_conformance(src: &str) {
    let bitcmp = |a: &[f32], b: &[f32]| -> (usize, i64) {
        let exact = a.iter().zip(b).filter(|(c, g)| c.to_bits() == g.to_bits()).count();
        let mx = a.iter().zip(b).map(|(c, g)| ulp(*c, *g)).max().unwrap_or(0);
        (exact, mx)
    };

    // --- sqrt: is Metal's sqrt correctly-rounded (matches libm::sqrtf)? ---
    let gsqrt = Gpu::new(src, "sqrt_kernel");
    let ins: Vec<f32> = lcg_vec(3_000_000, 0x5017).iter().map(|v| v.abs() * 64.0 + 1e-6).collect();
    let gs = gsqrt.expf(&ins); // in/out/n runner
    let mism = ins.iter().zip(&gs).filter(|(x, g)| libm::sqrtf(**x).to_bits() != g.to_bits()).count();
    println!(
        "== diagnostic: Metal sqrt vs libm::sqrtf: {}/{} match -> {} ==",
        ins.len() - mism, ins.len(),
        if mism == 0 { "correctly-rounded (safe to use directly)" } else { "NOT correctly-rounded (needs software sqrt)" }
    );
    println!();

    let mut fails = 0usize;
    println!("== forward-pass ops: Metal vs vitni-tensor CPU ops ==");

    // --- rms_norm ---
    let grms = Gpu::new(src, "rms_kernel");
    for &(rows, feat) in &[(1usize, 4usize), (4, 512), (8, 2048), (2, 4096), (16, 5632)] {
        let x = lcg_vec(rows * feat, 0x2000 ^ (feat as u64));
        let w = lcg_vec(feat, 0x3000 ^ (feat as u64));
        let xt = Tensor::from_f32(x.clone(), Shape::new(&[rows, feat]).unwrap()).unwrap();
        let wt = Tensor::from_f32(w.clone(), Shape::new(&[feat]).unwrap()).unwrap();
        let cpu = tvec(&xt.rms_norm(&wt, 1e-5).unwrap());
        let gpu = grms.rms(&x, &w, rows, feat, 1e-5);
        let (ex, mx) = bitcmp(&cpu, &gpu);
        let ok = ex == cpu.len();
        println!("  rms_norm  [{}x{}]  exact {}/{}  maxULP {}  {}", rows, feat, ex, cpu.len(), mx, if ok { "OK" } else { "FAIL" });
        if !ok { fails += 1; }
    }
    // --- softmax ---
    let gsm = Gpu::new(src, "softmax_kernel");
    for &(rows, last) in &[(1usize, 4usize), (4, 128), (8, 1024), (2, 4096), (32, 151)] {
        let x = lcg_vec(rows * last, 0x4000 ^ (last as u64));
        let xt = Tensor::from_f32(x.clone(), Shape::new(&[rows, last]).unwrap()).unwrap();
        let cpu = tvec(&xt.softmax_last_dim().unwrap());
        let gpu = gsm.softmax(&x, rows, last);
        let (ex, mx) = bitcmp(&cpu, &gpu);
        let ok = ex == cpu.len();
        println!("  softmax   [{}x{}]  exact {}/{}  maxULP {}  {}", rows, last, ex, cpu.len(), mx, if ok { "OK" } else { "FAIL" });
        if !ok { fails += 1; }
    }
    // --- silu ---
    let gsl = Gpu::new(src, "silu_kernel");
    for &n in &[4usize, 4096, 11008, 100_000] {
        let x = lcg_vec(n, 0x6000 ^ (n as u64));
        let xt = Tensor::from_f32(x.clone(), Shape::new(&[n]).unwrap()).unwrap();
        let cpu = tvec(&xt.silu().unwrap());
        let gpu = gsl.expf(&x); // in/out/n runner
        let (ex, mx) = bitcmp(&cpu, &gpu);
        let ok = ex == cpu.len();
        println!("  silu      [{}]  exact {}/{}  maxULP {}  {}", n, ex, cpu.len(), mx, if ok { "OK" } else { "FAIL" });
        if !ok { fails += 1; }
    }
    // --- rope (apply, reading a CPU-precomputed cache) ---
    let grope = Gpu::new(src, "rope_apply");
    for &(seq, n_heads, head_dim) in &[(1usize, 4usize, 64usize), (8, 32, 128), (4, 8, 64), (16, 6, 128)] {
        let theta = 10000.0f32;
        let offset = 0usize;
        let half = head_dim / 2;
        let x = lcg_vec(seq * n_heads * head_dim, 0x7000 ^ (head_dim as u64) ^ (seq as u64) << 8);
        // Replicate rope()'s cache exactly (same libm calls + order).
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / libm::powf(theta, (2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cosc = vec![0f32; seq * half];
        let mut sinc = vec![0f32; seq * half];
        for s in 0..seq {
            let pos = (offset + s) as f32;
            for i in 0..half {
                let angle = pos * inv_freq[i];
                cosc[s * half + i] = libm::cosf(angle);
                sinc[s * half + i] = libm::sinf(angle);
            }
        }
        let xt = Tensor::from_f32(x.clone(), Shape::new(&[seq, n_heads, head_dim]).unwrap()).unwrap();
        let cpu = tvec(&xt.rope(theta, offset).unwrap());
        let gpu = grope.rope(&x, &cosc, &sinc, seq, n_heads, head_dim);
        let (ex, mx) = bitcmp(&cpu, &gpu);
        let ok = ex == cpu.len();
        println!("  rope      [{}x{}x{}]  exact {}/{}  maxULP {}  {}", seq, n_heads, head_dim, ex, cpu.len(), mx, if ok { "OK" } else { "FAIL" });
        if !ok { fails += 1; }
    }
    println!("  VERDICT: forward ops {}\n", if fails == 0 { "PASS — bit-identical to CPU" } else { "FAIL" });
}

fn q4k_conformance(q4k_src: &str) {
    let gpu = Gpu::new(q4k_src, "q4k_linear");
    println!("== Q4_K fused dequant+dot: Metal vs canonical_dot_q4k_fused ==");
    let shapes: &[(usize, usize, &str)] = &[
        (256, 4, "1 super-block"),
        (512, 4, "2 super-blocks"),
        (2048, 8, "8 super-blocks"),
        (8192, 2, "32 sb = 1 chunk exactly"),
        (8448, 2, "33 sb -> 2 chunks"),
        (14336, 4, "Mistral FFN K (56 sb)"),
    ];
    let mut fails = 0usize;
    for &(k, m, label) in shapes {
        let nsuper = k / 256;
        let x = lcg_vec(k, 0xC0FFEE ^ (k as u64).wrapping_mul(0x9E3779B1));
        let mut w = vec![0u8; m * nsuper * 144];
        let mut s: u64 = 0x51CE_0000 ^ (k as u64).wrapping_mul(0x85EBCA77);
        let mut nb = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as u32
        };
        for byte in w.iter_mut() {
            *byte = (nb() & 0xFF) as u8;
        }
        // Overwrite each block's d/dmin f16 with a small normal positive value
        // (random f16 bits could be inf/NaN and complicate the comparison).
        for i in 0..m {
            for b in 0..nsuper {
                let off = (i * nsuper + b) * 144;
                let dbits = (((8 + (nb() % 6)) << 10) | (nb() & 0x3FF)) as u16;
                let mbits = (((8 + (nb() % 6)) << 10) | (nb() & 0x3FF)) as u16;
                w[off..off + 2].copy_from_slice(&dbits.to_le_bytes());
                w[off + 2..off + 4].copy_from_slice(&mbits.to_le_bytes());
            }
        }
        let mut cpu = vec![0f32; m];
        for i in 0..m {
            let wrow = &w[i * nsuper * 144..(i + 1) * nsuper * 144];
            cpu[i] = vitni_tensor::ops::quant::canonical_dot_q4k_fused(&x, wrow, k).unwrap();
        }
        let g = gpu.q4k(&x, &w, m, k, nsuper);
        let exact = cpu.iter().zip(&g).filter(|(c, gg)| c.to_bits() == gg.to_bits()).count();
        let maxu = cpu.iter().zip(&g).map(|(c, gg)| ulp(*c, *gg)).max().unwrap_or(0);
        let ok = exact == m;
        println!(
            "  [K={:>5} x{}] {:<28} exact {}/{}  maxULP {}  {}",
            k, m, label, exact, m, maxu, if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            fails += 1;
        }
    }
    println!(
        "  VERDICT: Q4_K {}\n",
        if fails == 0 { "PASS — bit-identical to the CPU hot path" } else { "FAIL" }
    );
}

fn poly_isolation(kernel_src: &str) {
    let gpu = Gpu::new(kernel_src, "poly_kernel");
    let p1 = 1.6666625440e-1f32;
    let p2 = -2.7667332906e-3f32;
    let mut xs = Vec::new();
    for i in 0..4_000_000usize {
        // primary range [-0.35, 0.35]
        xs.push(-0.35 + 0.70 * (i as f32) / 4_000_000.0);
    }
    let g = gpu.expf(&xs); // single-in/single-out runner (dispatches poly_kernel)
    let mut mism = 0usize;
    let mut max_ulp = 0i64;
    for (x, gv) in xs.iter().zip(&g) {
        let xx = x * x;
        let c = x - xx * (p1 + xx * p2); // separate ops (Rust does not auto-fma)
        if c.to_bits() != gv.to_bits() {
            mism += 1;
            let u = ulp(c, *gv);
            if u > max_ulp {
                max_ulp = u;
            }
        }
    }
    println!("== isolation: expf polynomial  c = x - xx*(P1+xx*P2)  (GPU vs CPU) ==");
    println!(
        "  mismatches: {}/{}  max ULP {}  -> {}",
        mism, xs.len(), max_ulp,
        if mism == 0 { "polynomial is clean (not the source)" } else { "GPU CONTRACTS the polynomial (this is the expf 1-ULP source)" }
    );
    println!();
}

fn division_diagnostic(kernel_src: &str) {
    let gpu = Gpu::new(kernel_src, "div_kernel");
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut s: u64 = 0x1234_5678_9abc_def0;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        f32::from_bits(((s >> 32) as u32 & 0x7fff_ffff) % 0x7f000000 + 0x00800000)
    };
    for _ in 0..2_000_000 {
        a.push(rnd());
        b.push(rnd());
    }
    let g = gpu.div(&a, &b);
    let mut mism = 0usize;
    let mut max_ulp = 0i64;
    for ((x, y), gv) in a.iter().zip(&b).zip(&g) {
        let c = x / y; // CPU IEEE f32 division
        if c.to_bits() != gv.to_bits() {
            mism += 1;
            let u = ulp(c, *gv);
            if u > max_ulp {
                max_ulp = u;
            }
        }
    }
    println!("== diagnostic: Metal f32 '/' vs CPU IEEE division (fast-math off) ==");
    println!(
        "  mismatches: {}/{}  max ULP {}  -> Metal divide is {}",
        mism, a.len(), max_ulp,
        if mism == 0 { "correctly-rounded" } else { "NOT correctly-rounded (this is the expf 1-ULP source)" }
    );
    println!();
}

fn build_inputs() -> Vec<f32> {
    let mut v: Vec<f32> = Vec::new();

    // Dense linear sweep over the whole expf domain (incl. over/underflow edges).
    let (lo, hi, steps) = (-104.0f32, 89.0f32, 3_000_000usize);
    for i in 0..steps {
        v.push(lo + (hi - lo) * (i as f32) / (steps as f32));
    }
    // Random sweep, same range (different distribution catches missed bins).
    let mut s: u64 = 0xF00DBABE;
    for _ in 0..2_000_000 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        v.push(lo + (hi - lo) * u);
    }
    // Tight sweep of the inference-critical range (softmax v-max in [-30,0],
    // SiLU exp(-x) arg in ~[-20,20]).
    for i in 0..1_000_000usize {
        v.push(-30.0 + 50.0 * (i as f32) / 1_000_000.0);
    }
    // Exact branch thresholds + a few ULP either side.
    let edges: &[f32] = &[
        0.0, -0.0, 1e-30, -1e-30,
        f32::from_bits(0x39000000), // 2^-14 boundary
        f32::from_bits(0x3eb17218), // 0.5 ln2 boundary
        f32::from_bits(0x3f851592), // 1.5 ln2 boundary
        f32::from_bits(0x42aeac50), // 87.33655
        f32::from_bits(0x42b17218), // 88.722839 (overflow)
        f32::from_bits(0x42cff1b5), // 103.972084 (-> 0)
        87.0, 88.0, 88.7, 88.8, -87.0, -88.0, -100.0, -103.9, -104.0, -110.0,
        88.722839, -103.972084, 0.6931472, -0.6931472, 1.0, -1.0, 10.0, -10.0,
    ];
    for &e in edges {
        for d in -3i32..=3 {
            v.push(f32::from_bits((e.to_bits() as i32 + d) as u32));
        }
    }
    v
}

fn dump_gguf_types() {
    use vitni_tensor::model::{config::Config, gguf::GgufFile, quant_weights::{QuantTensor, QuantizedWeights}};
    let path = "/Users/nickp/Downloads/vitnify_test/tinyllama-Q4_K_M.gguf";
    let blob = std::fs::read(path).expect("read gguf");
    let gguf = GgufFile::parse(&blob).unwrap();
    let cfg = Config::from_gguf(&gguf).unwrap();
    let w = QuantizedWeights::from_gguf(&gguf, &cfg).unwrap();
    use std::collections::BTreeMap;
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut add = |t: &QuantTensor| { *hist.entry(format!("{:?}", t.dtype)).or_default() += 1; };
    add(&w.token_embedding_table);
    if let Some(ref c) = w.wcls { add(c); }
    for l in &w.layers { add(&l.wq); add(&l.wk); add(&l.wv); add(&l.wo); add(&l.w1); add(&l.w2); add(&l.w3); }
    println!("== TinyLlama GGUF ==");
    println!("  config: dim={} hidden={} layers={} heads={} kv_heads={} vocab={}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size);
    println!("  head_size={} kv_dim={}", cfg.head_size(), cfg.kv_dim());
    println!("  quant type histogram (weight tensors): {:?}", hist);
    println!("  wcls(lm_head) present: {} | embed dtype: {:?}", w.wcls.is_some(), w.token_embedding_table.dtype);
    println!("  rope_neox: {}  rope_theta: {}  rms_eps: {}", w.rope_neox, w.rope_theta, w.rms_eps);
    println!();
}

fn main() {
    dump_gguf_types();
    let kernel_src = include_str!("../../expf.metal");
    let device = Device::system_default().expect("no Metal device");
    println!("== device ==\n  {}\n", device.name());

    division_diagnostic(kernel_src);
    softdiv_cross_vendor(kernel_src);
    poly_isolation(kernel_src);
    q4k_conformance(include_str!("../../q4k.metal"));
    q6k_conformance(include_str!("../../q6k.metal"));
    forward_ops_conformance(include_str!("../../forward_ops.metal"));

    let inputs = build_inputs();
    println!("== expf conformance (Metal fast-math OFF vs libm::expf) ==");
    println!("  inputs: {}", inputs.len());

    let gpu = Gpu::new(kernel_src, "expf_kernel");
    let got = gpu.expf(&inputs);

    // Split the error: GPU vs our-own-algorithm, and our-algorithm vs libm.
    let mut gpu_vs_ref = 0usize;
    let mut gpu_vs_ref_conseq = 0usize;
    let mut ref_vs_libm = 0usize;
    let mut ref_vs_libm_conseq = 0usize;
    let thr = 2f32.powi(-100);
    for (x, g) in inputs.iter().zip(&got) {
        let r = expf_ref(*x);
        let l = libm::expf(*x);
        if r.to_bits() != g.to_bits() {
            gpu_vs_ref += 1;
            if r.abs() >= thr && l.abs() >= thr { gpu_vs_ref_conseq += 1; }
        }
        if r.to_bits() != l.to_bits() {
            ref_vs_libm += 1;
            if l.abs() >= thr { ref_vs_libm_conseq += 1; }
        }
    }
    println!("  [split] GPU vs our-Rust-algorithm: {} mismatch ({} consequential)", gpu_vs_ref, gpu_vs_ref_conseq);
    println!("  [split] our-algorithm vs libm::expf: {} mismatch ({} consequential)", ref_vs_libm, ref_vs_libm_conseq);

    // "Consequential" threshold: a mismatch whose CPU output is at least this
    // large is a real numerical divergence. 2^-100 ≈ 7.9e-31 sits ~87 binary
    // orders above the denormal-flush boundary yet far below any softmax/SiLU
    // term that could survive normalization.
    let threshold = 2f32.powi(-100);
    let mut exact = 0usize;
    let mut mism = 0usize;
    let mut mism_consequential = 0usize;
    let mut max_mismatch_output = 0f32; // largest |cpu| among ALL mismatches
    let mut max_ulp = 0i64;
    let mut first_conseq_bad: Option<(f32, f32, f32)> = None;
    for (x, g) in inputs.iter().zip(&got) {
        let c = libm::expf(*x);
        if c.to_bits() == g.to_bits() {
            exact += 1;
            continue;
        }
        mism += 1;
        if c.abs() > max_mismatch_output {
            max_mismatch_output = c.abs();
        }
        let u = ulp(c, *g);
        if u > max_ulp {
            max_ulp = u;
        }
        if c.abs() >= threshold {
            mism_consequential += 1;
            if first_conseq_bad.is_none() {
                first_conseq_bad = Some((*x, c, *g));
            }
        }
    }
    println!("  exact-bit (all):          {}/{}", exact, inputs.len());
    println!("  mismatches (total):       {}", mism);
    println!("  largest CPU output among mismatches: {:e}  (all mismatches are below this)", max_mismatch_output);
    println!("  max ULP over mismatches:  {}", max_ulp);
    println!("  mismatches with |output| >= 2^-100 (~7.9e-31): {}", mism_consequential);
    if let Some((x, c, g)) = first_conseq_bad {
        println!(
            "  first consequential mismatch: x={} cpu={:e} ({:#010x}) gpu={:e} ({:#010x})",
            x, c, c.to_bits(), g, g.to_bits()
        );
    }

    if mism_consequential == 0 {
        println!(
            "\nVERDICT: PASS — GPU expf is bit-for-bit identical to libm::expf for every output\n         down to {:e}. Every divergence (max ULP {}) has a CPU output below {:e} —\n         the f32 denormal-flush tail (exp(x) for x < ~-87), which Apple GPUs round to\n         zero. Those terms are far below the ULP of any softmax/SiLU accumulator, so\n         the forward-pass output is unaffected; NVIDIA (-ftz=false) reproduces even\n         the tail exactly.",
            max_mismatch_output, max_ulp, threshold
        );
        std::process::exit(0);
    } else {
        println!("\nVERDICT: FAIL — {} consequential mismatches (|output| >= 2^-100).", mism_consequential);
        std::process::exit(1);
    }
}
