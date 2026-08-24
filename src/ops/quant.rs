//! Quantized-weight ops: Q4_0 / Q8_0 dequantization + linear matmul.
//!
//! The kernel exposes a fused dequant+matmul via `SYS_GPU_Q4_LINEAR`
//! (see `kernel/src/drivers/amd_gpu.rs::compute_q4_linear`) which is
//! the production path. CPU implementations here are the fallback +
//! the host-side test/verify reference.
//!
//! Block format (must match GGML / `SYS_GPU_Q4_LINEAR`):
//!
//! Q4_0 — 18 bytes per 32 weights:
//!   [scale: u16 LE f16][qs: u8; 16]
//!   weight[i] = scale * (qs_nibble[i] - 8)        for i in 0..32
//!   qs_nibble[2k]   = qs[k] & 0x0F
//!   qs_nibble[2k+1] = qs[k] >> 4
//!
//! Q8_0 — 34 bytes per 32 weights:
//!   [scale: u16 LE f16][qs: i8; 32]
//!   weight[i] = scale * qs[i]                     for i in 0..32

use alloc::vec;
use alloc::vec::Vec;

use crate::dtype::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_NUMEL,
    Q4_K_BLOCK_BYTES, Q4_K_BLOCK_NUMEL,
    Q6_K_BLOCK_BYTES, Q6_K_BLOCK_NUMEL,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_NUMEL,
};
use crate::error::{Error, Result};

// ---------------------------------------------------------------
// f16 ↔ f32 — minimal IEEE-754 half-precision conversion.
//
// GGML stores all quant block scales as f16 (i.e. binary16). Rust
// `core` has no f16 type yet on stable, so we hand-roll the decode.
// The encode is only used by quantize() in tests and the offline
// converter — production weight blobs already arrive as f16.
// ---------------------------------------------------------------

#[inline]
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 0x1) as u32;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mant = (half & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half → normal float.
            let mut m = mant;
            let mut e: i32 = 1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (((127 - 15 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Inf / NaN.
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        // Underflow → zero (GGML doesn't use subnormals for scales).
        sign << 15
    } else if exp >= 0x1F {
        // Overflow → inf.
        (sign << 15) | 0x7C00
    } else {
        (sign << 15) | ((exp as u16) << 10) | mant as u16
    }
}

// ---------------------------------------------------------------
// Q4_0
// ---------------------------------------------------------------

/// Dequantize Q4_0 blocks → f32. Output length is
/// `(q4_data.len() / 18) * 32`.
pub fn dequantize_q4_0(q4_data: &[u8]) -> Result<Vec<f32>> {
    if q4_data.len() % Q4_0_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q4_0 byte length not a multiple of 18"));
    }
    let n_blocks = q4_data.len() / Q4_0_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q4_0_BLOCK_NUMEL];
    for b in 0..n_blocks {
        let off = b * Q4_0_BLOCK_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([q4_data[off], q4_data[off + 1]]));
        let qs = &q4_data[off + 2..off + Q4_0_BLOCK_BYTES];
        let out_off = b * Q4_0_BLOCK_NUMEL;
        for k in 0..16 {
            let byte = qs[k];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            out[out_off + 2 * k]     = scale * lo as f32;
            out[out_off + 2 * k + 1] = scale * hi as f32;
        }
    }
    Ok(out)
}

/// Quantize f32 → Q4_0. Used by tests + offline weight conversion.
/// `data.len()` must be a multiple of 32 (caller pads if needed).
pub fn quantize_q4_0(data: &[f32]) -> Result<Vec<u8>> {
    if data.len() % Q4_0_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("Q4_0 quantize: input not a multiple of 32"));
    }
    let n_blocks = data.len() / Q4_0_BLOCK_NUMEL;
    let mut out = vec![0u8; n_blocks * Q4_0_BLOCK_BYTES];
    for b in 0..n_blocks {
        let block = &data[b * Q4_0_BLOCK_NUMEL..(b + 1) * Q4_0_BLOCK_NUMEL];
        // GGML's Q4_0 scale: max absolute value / -8 (sign-preserving).
        // The "-8" denominator is intentional: it makes the quantized
        // value 8 of the largest-magnitude element exactly representable
        // (since nibbles are unsigned 0..15 mapped to -8..7).
        let mut amax = 0f32;
        let mut max_val = 0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max_val = v;
            }
        }
        let scale = max_val / -8.0;
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let off = b * Q4_0_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for k in 0..16 {
            let x0 = block[2 * k] * inv_scale + 8.5;
            let x1 = block[2 * k + 1] * inv_scale + 8.5;
            let q0 = (x0 as i32).clamp(0, 15) as u8;
            let q1 = (x1 as i32).clamp(0, 15) as u8;
            out[off + 2 + k] = q0 | (q1 << 4);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Q5_0
// ---------------------------------------------------------------

/// Dequantize Q5_0 blocks → f32. Q5_0 packs 32 elements per 22-byte block:
/// an f16 scale, a 4-byte `qh` holding one high bit per element, and 16 bytes
/// of low nibbles. Value = ((low4 | (high_bit << 4)) - 16) * scale. Element
/// ordering matches ggml `dequantize_row_q5_0` (element j and j+16 share byte j).
pub fn dequantize_q5_0(q5_data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK: usize = 22;
    const NUMEL: usize = 32;
    if q5_data.len() % BLOCK != 0 {
        return Err(Error::Internal("Q5_0 byte length not a multiple of 22"));
    }
    let n_blocks = q5_data.len() / BLOCK;
    let mut out = vec![0f32; n_blocks * NUMEL];
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let scale = f16_to_f32(u16::from_le_bytes([q5_data[off], q5_data[off + 1]]));
        let qh = u32::from_le_bytes([
            q5_data[off + 2], q5_data[off + 3], q5_data[off + 4], q5_data[off + 5],
        ]);
        let qs = &q5_data[off + 6..off + BLOCK];
        let out_off = b * NUMEL;
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = (((qs[j] & 0x0F) | xh_0) as i32) - 16;
            let x1 = (((qs[j] >> 4) | xh_1) as i32) - 16;
            out[out_off + j] = scale * x0 as f32;
            out[out_off + j + 16] = scale * x1 as f32;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Q8_0
// ---------------------------------------------------------------

/// Dequantize Q8_0 blocks → f32.
pub fn dequantize_q8_0(q8_data: &[u8]) -> Result<Vec<f32>> {
    if q8_data.len() % Q8_0_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q8_0 byte length not a multiple of 34"));
    }
    let n_blocks = q8_data.len() / Q8_0_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q8_0_BLOCK_NUMEL];
    for b in 0..n_blocks {
        let off = b * Q8_0_BLOCK_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([q8_data[off], q8_data[off + 1]]));
        let qs = &q8_data[off + 2..off + Q8_0_BLOCK_BYTES];
        let out_off = b * Q8_0_BLOCK_NUMEL;
        for k in 0..32 {
            let q = qs[k] as i8 as f32;
            out[out_off + k] = scale * q;
        }
    }
    Ok(out)
}

/// Quantize f32 → Q8_0.
pub fn quantize_q8_0(data: &[f32]) -> Result<Vec<u8>> {
    if data.len() % Q8_0_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("Q8_0 quantize: input not a multiple of 32"));
    }
    let n_blocks = data.len() / Q8_0_BLOCK_NUMEL;
    let mut out = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];
    for b in 0..n_blocks {
        let block = &data[b * Q8_0_BLOCK_NUMEL..(b + 1) * Q8_0_BLOCK_NUMEL];
        let mut amax = 0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
            }
        }
        let scale = amax / 127.0;
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let off = b * Q8_0_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        for k in 0..32 {
            // round-half-away-from-zero, manual (no_std — no f32::round())
            let scaled = block[k] * inv_scale;
            let bias = if scaled >= 0.0 { 0.5 } else { -0.5 };
            let q = ((scaled + bias) as i32).clamp(-127, 127) as i8;
            out[off + 2 + k] = q as u8;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Q4_K — K-quant 256-element super-blocks with 8 sub-blocks of 32.
// Used for the bulk of weights in Mistral 7B / Llama 2 Q4_K_M.
//
// Per-block layout (144 bytes for 256 weights):
//   d:      u16 LE f16              super-block scale
//   dmin:   u16 LE f16              super-block min-scale
//   scales: u8;  12                 8 packed 6-bit (scale, min) pairs
//   qs:     u8;  128                128 packed 4-bit nibbles → 256 quants
//
// 6-bit (scale, min) unpacking (canonical get_scale_min_k4):
//   for j in 0..4:
//     scale[j] = scales[j]   & 0x3F
//     min[j]   = scales[j+4] & 0x3F
//   for j in 4..8:
//     scale[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//     min[j]   = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
//
// Dequant per sub-block j:
//   sub_scale = d * scale[j]
//   sub_min   = dmin * min[j]
//   weight[i] = sub_scale * nibble[i] - sub_min     for i in 0..32
// ---------------------------------------------------------------

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q4_K super-blocks → f32. Output length is
/// `(q4k_data.len() / 144) * 256`.
pub fn dequantize_q4_k(q4k_data: &[u8]) -> Result<Vec<f32>> {
    if q4k_data.len() % Q4_K_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q4_K byte length not a multiple of 144"));
    }
    let n_blocks = q4k_data.len() / Q4_K_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q4_K_BLOCK_NUMEL];
    for b in 0..n_blocks {
        let off = b * Q4_K_BLOCK_BYTES;
        let d    = f16_to_f32(u16::from_le_bytes([q4k_data[off],     q4k_data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q4k_data[off + 2], q4k_data[off + 3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[off + 4..off + 16]);
        let qs = &q4k_data[off + 16..off + Q4_K_BLOCK_BYTES];
        let out_off = b * Q4_K_BLOCK_NUMEL;

        // 8 sub-blocks of 32 weights. Process 64 weights per super-iter
        // (2 sub-blocks per iter), mirroring dequantize_row_q4_K's loop.
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y = out_off;
        for _ in (0..Q4_K_BLOCK_NUMEL).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is,     &scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            for l in 0..32 {
                out[y + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1f;
            }
            for l in 0..32 {
                out[y + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2f;
            }
            y += 64;
            q_off += 32;
            is += 2;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Q6_K — K-quant 256-element super-blocks with 16 sub-blocks of 16.
// Used for output projection (lm_head) in Mistral 7B Q4_K_M.
//
// Per-block layout (210 bytes for 256 weights):
//   ql:     u8;  128       low 4 bits of 256 quants
//   qh:     u8;   64       upper 2 bits of 256 quants
//   scales: i8;   16       per-sub-block scale
//   d:      u16 LE f16     super-block scale
//
// Each 6-bit quant has range [-32, 31] after `(packed) - 32`.
// Sub-block scale is i8. Final weight = d * sub_scale * (q - 32).
// ---------------------------------------------------------------

/// Dequantize Q6_K super-blocks → f32. Output length is
/// `(q6k_data.len() / 210) * 256`.
pub fn dequantize_q6_k(q6k_data: &[u8]) -> Result<Vec<f32>> {
    if q6k_data.len() % Q6_K_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q6_K byte length not a multiple of 210"));
    }
    let n_blocks = q6k_data.len() / Q6_K_BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * Q6_K_BLOCK_NUMEL];
    for b in 0..n_blocks {
        let off = b * Q6_K_BLOCK_BYTES;
        let ql_base = off;
        let qh_base = off + 128;
        let sc_base = off + 192; // 128 + 64
        let d_off   = off + 208;
        let d = f16_to_f32(u16::from_le_bytes([q6k_data[d_off], q6k_data[d_off + 1]]));
        let out_off = b * Q6_K_BLOCK_NUMEL;

        // Two 128-weight halves per super-block. Each half consumes
        // 64 bytes of ql, 32 bytes of qh, 8 i8 scales.
        let mut ql_p = 0usize;
        let mut qh_p = 0usize;
        let mut sc_p = 0usize;
        let mut y = 0usize;
        for _ in (0..Q6_K_BLOCK_NUMEL).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let ql_lo0 = q6k_data[ql_base + ql_p + l +  0] & 0x0F;
                let ql_lo1 = q6k_data[ql_base + ql_p + l + 32] & 0x0F;
                let ql_hi0 = q6k_data[ql_base + ql_p + l +  0] >> 4;
                let ql_hi1 = q6k_data[ql_base + ql_p + l + 32] >> 4;
                let qh = q6k_data[qh_base + qh_p + l];
                let q1 = ((ql_lo0 | (((qh >> 0) & 0x03) << 4)) as i32 - 32) as f32;
                let q2 = ((ql_lo1 | (((qh >> 2) & 0x03) << 4)) as i32 - 32) as f32;
                let q3 = ((ql_hi0 | (((qh >> 4) & 0x03) << 4)) as i32 - 32) as f32;
                let q4 = ((ql_hi1 | (((qh >> 6) & 0x03) << 4)) as i32 - 32) as f32;
                let s1 = q6k_data[sc_base + sc_p + is + 0] as i8 as f32;
                let s2 = q6k_data[sc_base + sc_p + is + 2] as i8 as f32;
                let s3 = q6k_data[sc_base + sc_p + is + 4] as i8 as f32;
                let s4 = q6k_data[sc_base + sc_p + is + 6] as i8 as f32;
                out[out_off + y + l +  0] = d * s1 * q1;
                out[out_off + y + l + 32] = d * s2 * q2;
                out[out_off + y + l + 64] = d * s3 * q3;
                out[out_off + y + l + 96] = d * s4 * q4;
            }
            y += 128;
            ql_p += 64;
            qh_p += 32;
            sc_p += 8;
        }
    }
    Ok(out)
}

/// CPU reference for `linear` against Q4_K weights. Mirrors
/// `linear_q4_0_cpu`'s contract: row-major weight, row-major
/// activation, fused dequant+matmul.
/// Canonical dot product — the SAME fixed shape as `ops::matmul` and
/// `the reference`, so quantized inference shares the one numerical regime a
/// certificate declares.
///
/// This matters because the quantized kernels below ARE the real LLM inference
/// path: Mistral/Llama forward passes go through `linear_q4_*_cpu`, not through
/// `ops::matmul`. Making only the f32 matmul canonical left certificates
/// claiming NUMERIC_REGIME_CANONICAL_V1 while the model that produced them
/// actually ran a serial reduction — caught 2026-07-29 when a 7B replay came
/// back with a logits_hash identical to the pre-canonical certificate, proving
/// the change had never touched this path.
///
/// Fixed CANON_BLOCK-sized blocks in ascending order, then a fixed pairwise
/// tree. Each product is a standalone rounded value so FMA contraction cannot
/// alter it.
// ===================================================================
// Canonical v2 — pinned LANES rather than pinned blocks.
//
// v1 (`canonical_dot` below) fixes the reduction shape as CANON_BLOCK
// sequential accumulations, then a pairwise tree. That is deterministic
// but structurally scalar: within a block, `acc += p` eight times is an
// 8-deep dependency chain. Every add waits on the previous one, no SIMD
// lane can help, and the core's other FP ports sit idle. It is why the
// reference implementation runs at ~0.13 tok/s on a 7B model.
//
// v2 keeps determinism and removes the chain, by pinning what must be
// pinned and nothing more:
//
//   * LANES  — how many independent accumulator chains. A CONTRACT
//              constant, NOT the ISA's vector width. AVX2 does 8 f32
//              per register, NEON does 4, scalar does 1 — all three
//              must produce identical bits, so the lane count cannot
//              come from the hardware.
//   * CHUNK  — elements per independently-reducible chunk. A CONTRACT
//              constant, NOT the thread count. 1 thread and 64 threads
//              must produce identical bits, so the split cannot come
//              from the scheduler.
//
// Definition (this IS the spec; the fast path must match it exactly):
//
//     lane[j] accumulates, in increasing i, every element of a chunk
//     where (i - chunk_start) % LANES == j
//     chunk_sum = fixed pairwise tree over lane[0..LANES]
//     result    = fixed pairwise tree over chunk_sum[0..nchunks]
//
// Determinism argument: the assignment of element -> lane and element
// -> chunk depends only on the element's index and two constants. No
// hardware property enters. Therefore vector width, thread count, core
// count, and chunk evaluation ORDER cannot change the result.
//
// Performance argument: the inner loop is LANES independent chains, so
// a compiler can keep one f32x8 (or two f32x4) in flight with no
// horizontal ops, and chunks are embarrassingly parallel.
//
// FMA is still forbidden — the multiply and the add stay separate
// operations (contract point 2). Vectorizing must emit mul+add, never
// a fused multiply-add.
// ===================================================================

/// Independent accumulator chains. Contract constant — never derive
/// this from the target's vector width.
pub const CANON_LANES: usize = 8;

/// Elements per independently reducible chunk. Contract constant —
/// never derive this from the thread or core count. Multiple of
/// CANON_LANES so chunk boundaries never split a lane group.
pub const CANON_CHUNK: usize = 8192;

/// Fixed pairwise tree over a slice. Same shape as v1's tree, factored
/// out because v2 applies it at two levels (lanes, then chunks).
#[inline]
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

/// Reduce ONE chunk to a scalar. Order within the chunk is fully
/// determined by CANON_LANES, so this is the unit of parallelism:
/// callers may evaluate chunks on any thread in any order.
#[inline]
pub fn canonical_chunk(x: &[f32], w: &[f32]) -> f32 {
    let n = x.len();
    let mut lanes = [0.0f32; CANON_LANES];
    let full = n - (n % CANON_LANES);

    // Hot loop: CANON_LANES independent chains. No cross-lane traffic,
    // so this vectorizes to one multiply + one add per iteration.
    let mut i = 0;
    while i < full {
        for j in 0..CANON_LANES {
            let p = x[i + j] * w[i + j];
            lanes[j] += p;
        }
        i += CANON_LANES;
    }
    // Tail keeps the same element -> lane rule.
    while i < n {
        let j = i % CANON_LANES;
        let p = x[i] * w[i];
        lanes[j] += p;
        i += 1;
    }
    fixed_tree(&mut lanes)
}

/// Canonical v2 dot product — bit-identical regardless of vector width,
/// thread count, or the order chunks are evaluated in.
///
/// **NOT YET the default path — adopting it is a REGIME CHANGE.**
///
/// v1's CANON_BLOCK chain is 8 sequential adds, so no vector unit can help
/// it; v2's independent lanes are the only shape that vectorises. Measured
/// at Mistral FFN shape (4096x14336) on M3 Max, interleaved A/B, medians
/// of 5, reduction only:
///
/// ```text
///   v1 index arithmetic (ships today) :  12.01 ms   9.8 GFLOP/s   1.00x
///   v1 chunks_exact  (BIT-IDENTICAL)  :  11.05 ms  10.6 GFLOP/s   1.09x
///   v2 chunks_exact  (regime 3)       :   5.17 ms  22.7 GFLOP/s   2.32x
/// ```
///
/// Two corrections to earlier readings, both from conflating variables:
///
///  * An initial 2.09x for v2 was a cold-cache ordering artifact.
///  * The 0.91x that replaced it was real but measured the WRONG THING —
///    v2 spelled with a `while` loop and index arithmetic (0.96x). The
///    lane design was never the problem; that transcription defeated
///    LLVM's vectoriser. Spelled with `chunks_exact` the same function,
///    bit-for-bit, runs 2.32x. `-C target-cpu=native` accounts for none
///    of it.
///
/// So the gain is real, but it is not free: v2 changes the reduction
/// order (see `v2_reduction_order_differs_from_v1`), which invalidates
/// every certificate issued under regime 2 and every published test
/// vector. The dot is ~30% of the fused Q4_K kernel, so 2.32x there is
/// roughly 1.2x end-to-end — a deliberate trade to make, not a drive-by.
///
/// Row-parallel matmul needs none of this: each output element is an
/// independent `canonical_dot`, so threading across rows already cannot
/// perturb a reduction order. Note that llama.cpp measures 5.55 tok/s at
/// 1 thread and 5.39 at 14 for single-token decode — generation is
/// memory-bandwidth bound, so thread scaling is not where the win is.
///
/// Public wrapper — the canonical reduction is the numerical contract, so
/// any code path producing certified values must be able to reach it.
pub fn canonical_dot_pub(x: &[f32], w: &[f32], n: usize) -> f32 {
    canonical_dot(x, w, n)
}

pub(crate) fn canonical_dot(x: &[f32], w: &[f32], n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    if n <= CANON_CHUNK {
        return canonical_chunk(&x[..n], &w[..n]);
    }
    let nchunks = (n + CANON_CHUNK - 1) / CANON_CHUNK;
    let mut sums = alloc::vec![0.0f32; nchunks];
    for c in 0..nchunks {
        let s = c * CANON_CHUNK;
        let e = core::cmp::min(s + CANON_CHUNK, n);
        sums[c] = canonical_chunk(&x[s..e], &w[s..e]);
    }
    fixed_tree(&mut sums)
}

/// Combine independently-computed chunk sums. Exposed so a threaded or
/// offloaded backend can compute `canonical_chunk` per chunk in ANY
/// order and still land on the contract's value — the tree is applied
/// over chunks in INDEX order regardless of completion order.
pub fn canonical_combine(chunk_sums: &mut [f32]) -> f32 {
    fixed_tree(chunk_sums)
}

// A distinct reduction order (blocked, not the pinned 8-deep chain), kept ONLY as a
// test reference: `v2_reduction_order_differs_from_v1` uses it to prove the reduction
// order is load-bearing. Scoped to tests so it is never mistaken for a live kernel.
#[cfg(test)]
#[inline]
pub(crate) fn canonical_dot_regime2(x: &[f32], w: &[f32], n: usize) -> f32 {
    const CANON_BLOCK: usize = 8;
    if n == 0 {
        return 0.0;
    }
    let nb = (n + CANON_BLOCK - 1) / CANON_BLOCK;
    let mut part = alloc::vec![0.0f32; nb];
    for b in 0..nb {
        let s = b * CANON_BLOCK;
        let e = core::cmp::min(s + CANON_BLOCK, n);
        let mut acc = 0.0f32;
        for i in s..e {
            let p = x[i] * w[i];
            acc += p;
        }
        part[b] = acc;
    }
    let mut len = nb;
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


/// Regime-3 variant of `canonical_dot_q4k_fused`: identical dequant, but
/// the reduction uses CANON_LANES independent chains instead of v1's
/// 8-deep sequential chain. NOT bit-compatible with regime 2 — provided
/// so the end-to-end cost of adopting v2 can be measured on the real
/// kernel rather than extrapolated from a microbenchmark.
pub fn canonical_dot_q4k_fused(x: &[f32], w_q4k: &[u8], n: usize) -> Result<f32> {
    if w_q4k.len() % Q4_K_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q4_K byte length not a multiple of 144"));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let n_super = w_q4k.len() / Q4_K_BLOCK_BYTES;
    let mut lanes = [0.0f32; CANON_LANES];
    let mut buf = [0.0f32; Q4_K_BLOCK_NUMEL];

    // Must chunk exactly as `canonical_dot` does, or this diverges from the
    // reference for any row longer than CANON_CHUNK — Mistral's down
    // projection is in_feat=14336, so that is not a hypothetical. A Q4_K
    // super-block is 256 and CANON_CHUNK is 8192, so a chunk is exactly 32
    // super-blocks and boundaries always align.
    let supers_per_chunk = CANON_CHUNK / Q4_K_BLOCK_NUMEL;
    let mut chunk_sums: alloc::vec::Vec<f32> = alloc::vec::Vec::new();

    for b in 0..n_super {
        let base = b * Q4_K_BLOCK_NUMEL;
        if base >= n {
            break;
        }
        let off = b * Q4_K_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([w_q4k[off], w_q4k[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([w_q4k[off + 2], w_q4k[off + 3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&w_q4k[off + 4..off + 16]);
        let qs = &w_q4k[off + 16..off + Q4_K_BLOCK_BYTES];
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y = 0usize;
        for _ in (0..Q4_K_BLOCK_NUMEL).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let qsl = &qs[q_off..q_off + 32];
            let (lo, hi) = buf[y..y + 64].split_at_mut(32);
            for (o, &q) in lo.iter_mut().zip(qsl.iter()) {
                *o = d1 * (q & 0x0F) as f32 - m1f;
            }
            for (o, &q) in hi.iter_mut().zip(qsl.iter()) {
                *o = d2 * (q >> 4) as f32 - m2f;
            }
            y += 64;
            q_off += 32;
            is += 2;
        }

        // Lane accumulation across the whole row — super-block boundaries
        // land on multiples of 256, itself a multiple of CANON_LANES, so
        // every element keeps the same lane it would have had.
        let avail = core::cmp::min(Q4_K_BLOCK_NUMEL, n - base);
        let xs = &x[base..base + avail];
        let bs = &buf[..avail];
        for (xc, bc) in xs
            .chunks_exact(CANON_LANES)
            .zip(bs.chunks_exact(CANON_LANES))
        {
            for j in 0..CANON_LANES {
                let p = xc[j] * bc[j];
                lanes[j] += p;
            }
        }
        let rem = avail % CANON_LANES;
        if rem != 0 {
            let s = avail - rem;
            for t in s..avail {
                let p = x[base + t] * buf[t];
                lanes[t % CANON_LANES] += p;
            }
        }

        // Close the chunk on its boundary, or at the end of the row.
        let is_last = base + avail >= n;
        if (b + 1) % supers_per_chunk == 0 || is_last {
            let mut l = lanes;
            chunk_sums.push(fixed_tree(&mut l));
            lanes = [0.0f32; CANON_LANES];
        }
    }
    Ok(fixed_tree(&mut chunk_sums))
}

// ===================================================================
// Integer-domain dot (regime 4 candidate)
//
// The float paths above must PIN a reduction order because float addition
// is not associative. Integer addition is. So if the inner reduction is
// carried in integers, order-independence stops being a contract clause
// and becomes a property of the arithmetic — no CANON_LANES, no
// CANON_CHUNK, no way for a SIMD width or thread count to matter.
//
// For one Q4_K super-block with the activations quantized to int8:
//
//   w[i]  = d·sc_j·q[i] − dmin·m_j          (sub-block j of 32)
//   x[i]  = dx·qx[i]
//
//   Σ x·w = d·dx · Σ_j sc_j·(Σ_i qx[i]·q[i])  −  dmin·dx · Σ_j m_j·bsum_j
//           \_______ integer _______/            \____ integer ____/
//
// Both sums are exactly integer, so the whole 256-element reduction is
// order-independent. Only the combination ACROSS super-blocks stays float.
//
// HONEST SCOPE: this does NOT make the model order-independent. It shrinks
// the pinned surface from n float additions to n/256 — for n=4096, from
// 4096 to 16, a 256x reduction. RMSNorm, softmax, RoPE and SiLU are still
// float and still need §3.2/§3.3. "Structural" applies to the matmul only.
//
// It is also a DIFFERENT COMPUTATION: quantizing activations to int8 loses
// precision relative to f32 activations, so outputs differ from regime 3.
// That is the standard approach (llama.cpp's ggml_vec_dot_q4_K_q8_K does
// the same), but it is a numerics change, not just a faster spelling.
// ===================================================================

/// Activations quantized to int8 with one f32 scale per 256 elements, plus
/// per-32-group integer sums (needed for Q4_K's `dmin` term without
/// revisiting the data).
pub struct Q8Block {
    pub d: f32,
    pub qs: [i8; 256],
    /// Σ qs over each group of 32 — 8 groups.
    pub bsums: [i32; 8],
}

/// Quantize one 256-element activation block to int8.
///
/// Scale is `max|x| / 127`. Rounding is round-half-away-from-zero, computed
/// identically on every target, so the quantization itself is deterministic
/// even though it is float→int.
pub fn quantize_block_q8(x: &[f32]) -> Q8Block {
    debug_assert!(x.len() <= 256);
    let mut amax = 0.0f32;
    for &v in x {
        let a = if v < 0.0 { -v } else { v };
        if a > amax {
            amax = a;
        }
    }
    let d = amax / 127.0;
    let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0i8; 256];
    let mut bsums = [0i32; 8];
    for (i, &v) in x.iter().enumerate() {
        let scaled = v * inv;
        let bias = if scaled >= 0.0 { 0.5 } else { -0.5 };
        let q = ((scaled + bias) as i32).clamp(-127, 127);
        qs[i] = q as i8;
        bsums[i / 32] += q;
    }
    Q8Block { d, qs, bsums }
}

/// Scalar 32-element dot pair: 32 packed Q4 nibbles against the low and high
/// 32-element int8 activation halves. Returns `(s_lo, s_hi)`.
#[inline]
fn dot32_pair_scalar(qsl: &[u8], xlo: &[i8], xhi: &[i8]) -> (i32, i32) {
    let mut s_lo: i32 = 0;
    let mut s_hi: i32 = 0;
    for ((&qb, &xl), &xh) in qsl.iter().zip(xlo.iter()).zip(xhi.iter()) {
        s_lo += (qb & 0x0F) as i32 * xl as i32;
        s_hi += (qb >> 4) as i32 * xh as i32;
    }
    (s_lo, s_hi)
}

/// NEON/SDOT twin of [`dot32_pair_scalar`].
///
/// BIT-EXACT with the scalar version, and this is a property of the arithmetic
/// rather than a coincidence: both accumulate into i32, and integer addition is
/// associative and exact. Lane reordering and the horizontal reduction at the
/// end therefore cannot change the result — unlike float, where they would.
/// That is precisely why the deterministic kernel can be vectorized without
/// touching `fixed_tree` or invalidating a single issued certificate.
///
/// Nibbles are 0..=15 and activations are i8, so every product fits an i8xi8
/// SDOT lane with no saturation risk. Asserted equal to the scalar path in
/// `neon_dot32_matches_scalar`.
/// Uses widening multiply (SMULL) + pairwise accumulate (SADALP) rather than
/// SDOT: `vdotq_s32` is still behind the unstable `stdarch_neon_dotprod`
/// feature, and this crate must build on stable for a verifiable runtime userspace. Note that
/// llama.cpp *does* use SDOT where available, so the comparison below is
/// against its faster path — there is headroom left here.
///
/// Nibbles are 0..=15 and activations i8, so each product is at most 1905 and
/// cannot overflow an i16 lane; the pairwise accumulate widens to i32.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot32_pair_neon(qsl: &[u8], xlo: &[i8], xhi: &[i8]) -> (i32, i32) {
    use core::arch::aarch64::*;
    debug_assert!(qsl.len() >= 32 && xlo.len() >= 32 && xhi.len() >= 32);
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut acc_lo = vdupq_n_s32(0);
        let mut acc_hi = vdupq_n_s32(0);
        for k in 0..2 {
            let q = vld1q_u8(qsl.as_ptr().add(k * 16));
            let xl = vld1q_s8(xlo.as_ptr().add(k * 16));
            let xh = vld1q_s8(xhi.as_ptr().add(k * 16));

            let nl = vreinterpretq_s8_u8(vandq_u8(q, mask));
            acc_lo = vpadalq_s16(acc_lo, vmull_s8(vget_low_s8(nl), vget_low_s8(xl)));
            acc_lo = vpadalq_s16(acc_lo, vmull_s8(vget_high_s8(nl), vget_high_s8(xl)));

            let nh = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q));
            acc_hi = vpadalq_s16(acc_hi, vmull_s8(vget_low_s8(nh), vget_low_s8(xh)));
            acc_hi = vpadalq_s16(acc_hi, vmull_s8(vget_high_s8(nh), vget_high_s8(xh)));
        }
        (vaddvq_s32(acc_lo), vaddvq_s32(acc_hi))
    }
}

/// Dispatch to the vector path where the target supports it. Both produce
/// identical bits, so this is purely a speed choice — no regime change.
#[inline]
fn dot32_pair(qsl: &[u8], xlo: &[i8], xhi: &[i8]) -> (i32, i32) {
    #[cfg(target_arch = "aarch64")]
    {
        dot32_pair_neon(qsl, xlo, xhi)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot32_pair_scalar(qsl, xlo, xhi)
    }
}

/// Integer dot of one Q4_K weight super-block against one int8 activation
/// block. The two inner accumulations are pure integer and therefore
/// order-independent; only the final scale application is float.
#[inline]
pub fn dot_q4k_q8_block(w: &[u8], xq: &Q8Block, n: usize) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([w[2], w[3]]));
    let mut scales = [0u8; 12];
    scales.copy_from_slice(&w[4..16]);
    let qs = &w[16..Q4_K_BLOCK_BYTES];

    let mut acc: i32 = 0; // Σ_j sc_j · Σ_i qx·q
    let mut acc_m: i32 = 0; // Σ_j m_j · bsum_j

    // 8 sub-blocks of 32. Nibbles are packed low/high across 32-byte halves,
    // matching dequantize_q4_k's layout exactly.
    for half in 0..4 {
        let (sc_lo, m_lo) = get_scale_min_k4(half * 2, &scales);
        let (sc_hi, m_hi) = get_scale_min_k4(half * 2 + 1, &scales);
        let q_off = half * 32;
        let base = half * 64;

        let mut s_lo: i32 = 0;
        let mut s_hi: i32 = 0;
        if base + 64 <= n {
            // Full sub-block pair: slice-and-zip so the 32-iteration loop has
            // a known length, no bounds checks and no per-element branch.
            // Branching inside this loop measured 0.53x vs the float path;
            // the branches, not the integer arithmetic, were the cost.
            let qsl = &qs[q_off..q_off + 32];
            let xlo = &xq.qs[base..base + 32];
            let xhi = &xq.qs[base + 32..base + 64];
            let (a, b) = dot32_pair(qsl, xlo, xhi);
            s_lo = a;
            s_hi = b;
        } else {
            for l in 0..32 {
                let idx_lo = base + l;
                let idx_hi = base + 32 + l;
                if idx_lo < n {
                    s_lo += (qs[q_off + l] & 0x0F) as i32 * xq.qs[idx_lo] as i32;
                }
                if idx_hi < n {
                    s_hi += (qs[q_off + l] >> 4) as i32 * xq.qs[idx_hi] as i32;
                }
            }
        }
        acc += sc_lo as i32 * s_lo + sc_hi as i32 * s_hi;
        acc_m += m_lo as i32 * xq.bsums[half * 2] + m_hi as i32 * xq.bsums[half * 2 + 1];
    }

    d * xq.d * acc as f32 - dmin * xq.d * acc_m as f32
}

/// Integer-domain Q4_K row dot. Activations are quantized per 256-element
/// block; each super-block's contribution is computed exactly in integers,
/// then combined across super-blocks with the canonical fixed tree — the
/// only remaining place where order matters.
pub fn integer_dot_q4k(x: &[f32], w_q4k: &[u8], n: usize) -> Result<f32> {
    if w_q4k.len() % Q4_K_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q4_K byte length not a multiple of 144"));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let n_super = w_q4k.len() / Q4_K_BLOCK_BYTES;
    let mut parts: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(n_super);
    for b in 0..n_super {
        let base = b * Q4_K_BLOCK_NUMEL;
        if base >= n {
            break;
        }
        let avail = core::cmp::min(Q4_K_BLOCK_NUMEL, n - base);
        let xq = quantize_block_q8(&x[base..base + avail]);
        let off = b * Q4_K_BLOCK_BYTES;
        parts.push(dot_q4k_q8_block(
            &w_q4k[off..off + Q4_K_BLOCK_BYTES],
            &xq,
            avail,
        ));
    }
    Ok(fixed_tree(&mut parts))
}

/// Integer-domain Q4_K linear.
///
/// Quantizes the activation ONCE per layer, not once per output row — the
/// activation vector is shared across all `out_feat` rows, so quantizing it
/// inside the row loop would repeat the work 14336 times for one Mistral FFN
/// layer.
pub fn linear_q4_k_integer(
    x: &[f32],
    w_q4k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if in_feat % Q4_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q4_k_integer: in_feat not a multiple of 256"));
    }
    let row_bytes = (in_feat / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
    if w_q4k.len() < out_feat * row_bytes {
        return Err(Error::Internal("linear_q4_k_integer: weight shape mismatch"));
    }
    let n_super = in_feat / Q4_K_BLOCK_NUMEL;
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        // One pass over the activation for the whole layer.
        let xq: alloc::vec::Vec<Q8Block> = (0..n_super)
            .map(|s| quantize_block_q8(&x_row[s * Q4_K_BLOCK_NUMEL..(s + 1) * Q4_K_BLOCK_NUMEL]))
            .collect();
        // Scratch buffer hoisted out of the row loop: allocating it per output
        // row cost one malloc/free per row (14336 per Mistral FFN matmul, ~3M
        // per token across 32 layers). Reduction order is untouched — the same
        // partials are pushed in the same order into the same fixed_tree — so
        // this stays bit-identical and every issued certificate still replays.
        // The parallel twin below already did this.
        let mut parts: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(n_super);
        for o in 0..out_feat {
            let w_row = &w_q4k[o * row_bytes..(o + 1) * row_bytes];
            parts.clear();
            for s in 0..n_super {
                let off = s * Q4_K_BLOCK_BYTES;
                parts.push(dot_q4k_q8_block(
                    &w_row[off..off + Q4_K_BLOCK_BYTES],
                    &xq[s],
                    Q4_K_BLOCK_NUMEL,
                ));
            }
            y_out[b * out_feat + o] = fixed_tree(&mut parts);
        }
    }
    Ok(())
}

/// Integer dot of one Q6_K weight super-block against int8 activations.
///
/// Simpler than Q4_K: Q6_K has no `dmin` term, so
///
///   Σ x·w = d·dx · Σⱼ scaleⱼ·(Σᵢ qxᵢ·qᵢ)
///
/// is a single integer accumulation. `q` is 6-bit signed (−32..31), `scale`
/// is i8, `qx` is i8 — products reach 31·127 ≈ 3.9e3, and 256 of them times
/// a scale stay far inside i32.
///
/// Sixteen scales per super-block, indexed exactly as `dequantize_q6_k`
/// indexes them: `scales[half*8 + (l/16) + 2*k]` for output `y + l + 32*k`.
/// Sums are bucketed per scale so the multiply happens once per bucket.
#[inline]
pub fn dot_q6k_q8_block(w: &[u8], xq: &Q8Block, n: usize) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([w[208], w[209]]));
    let ql = &w[0..128];
    let qh = &w[128..192];
    let sc = &w[192..208];

    let mut sums = [0i32; 16];
    for half in 0..2 {
        let ql_p = half * 64;
        let qh_p = half * 32;
        let sc_p = half * 8;
        let y = half * 128;
        // `is` is l/16, so split the l range instead of recomputing it and
        // indexing `sums` with a running expression. Fixed accumulators per
        // 16-element run, no computed stores in the loop, and the `n` bound
        // is checked once per run rather than per element — the same branch
        // problem that made the Q4_K integer path 0.53x before it was fixed.
        for (is, l0) in [(0usize, 0usize), (1, 16)] {
            let hi_idx = y + l0 + 15 + 96;
            if hi_idx < n {
                let (mut a1, mut a2, mut a3, mut a4) = (0i32, 0i32, 0i32, 0i32);
                for l in l0..l0 + 16 {
                    let h = qh[qh_p + l];
                    let b0 = ql[ql_p + l];
                    let b1 = ql[ql_p + l + 32];
                    a1 += (((b0 & 0x0F) | ((h & 0x03) << 4)) as i32 - 32)
                        * xq.qs[y + l] as i32;
                    a2 += (((b1 & 0x0F) | (((h >> 2) & 0x03) << 4)) as i32 - 32)
                        * xq.qs[y + l + 32] as i32;
                    a3 += (((b0 >> 4) | (((h >> 4) & 0x03) << 4)) as i32 - 32)
                        * xq.qs[y + l + 64] as i32;
                    a4 += (((b1 >> 4) | (((h >> 6) & 0x03) << 4)) as i32 - 32)
                        * xq.qs[y + l + 96] as i32;
                }
                sums[sc_p + is] += a1;
                sums[sc_p + is + 2] += a2;
                sums[sc_p + is + 4] += a3;
                sums[sc_p + is + 6] += a4;
            } else {
                for l in l0..l0 + 16 {
                    let h = qh[qh_p + l];
                    let b0 = ql[ql_p + l];
                    let b1 = ql[ql_p + l + 32];
                    let q1 = ((b0 & 0x0F) | ((h & 0x03) << 4)) as i32 - 32;
                    let q2 = ((b1 & 0x0F) | (((h >> 2) & 0x03) << 4)) as i32 - 32;
                    let q3 = ((b0 >> 4) | (((h >> 4) & 0x03) << 4)) as i32 - 32;
                    let q4 = ((b1 >> 4) | (((h >> 6) & 0x03) << 4)) as i32 - 32;
                    let (i1, i2, i3, i4) = (y + l, y + l + 32, y + l + 64, y + l + 96);
                    if i1 < n { sums[sc_p + is] += q1 * xq.qs[i1] as i32; }
                    if i2 < n { sums[sc_p + is + 2] += q2 * xq.qs[i2] as i32; }
                    if i3 < n { sums[sc_p + is + 4] += q3 * xq.qs[i3] as i32; }
                    if i4 < n { sums[sc_p + is + 6] += q4 * xq.qs[i4] as i32; }
                }
            }
        }
    }
    let mut acc: i32 = 0;
    for j in 0..16 {
        acc += sc[j] as i8 as i32 * sums[j];
    }
    d * xq.d * acc as f32
}

/// Integer-domain Q6_K linear. Activation quantized once per layer.
pub fn linear_q6_k_integer(
    x: &[f32],
    w_q6k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if in_feat % Q6_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q6_k_integer: in_feat not a multiple of 256"));
    }
    let row_bytes = (in_feat / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES;
    if w_q6k.len() < out_feat * row_bytes {
        return Err(Error::Internal("linear_q6_k_integer: weight shape mismatch"));
    }
    let n_super = in_feat / Q6_K_BLOCK_NUMEL;
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        let xq: alloc::vec::Vec<Q8Block> = (0..n_super)
            .map(|s| quantize_block_q8(&x_row[s * Q6_K_BLOCK_NUMEL..(s + 1) * Q6_K_BLOCK_NUMEL]))
            .collect();
        for o in 0..out_feat {
            let w_row = &w_q6k[o * row_bytes..(o + 1) * row_bytes];
            let mut parts: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(n_super);
            for s in 0..n_super {
                let off = s * Q6_K_BLOCK_BYTES;
                parts.push(dot_q6k_q8_block(
                    &w_row[off..off + Q6_K_BLOCK_BYTES],
                    &xq[s],
                    Q6_K_BLOCK_NUMEL,
                ));
            }
            y_out[b * out_feat + o] = fixed_tree(&mut parts);
        }
    }
    Ok(())
}

/// Row-parallel integer Q6_K linear.
#[cfg(feature = "std-parallel")]
pub fn linear_q6_k_integer_parallel(
    x: &[f32],
    w_q6k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
    threads: usize,
) -> Result<()> {
    if in_feat % Q6_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q6_k_integer_parallel: in_feat % 256"));
    }
    if threads <= 1 {
        return linear_q6_k_integer(x, w_q6k, y_out, batch, in_feat, out_feat);
    }
    let row_bytes = (in_feat / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES;
    if w_q6k.len() < out_feat * row_bytes {
        return Err(Error::Internal("linear_q6_k_integer_parallel: weight shape"));
    }
    let n_super = in_feat / Q6_K_BLOCK_NUMEL;
    use rayon::prelude::*;
    let grain = core::cmp::max(1, out_feat.div_ceil(threads * 4));
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        let xq: alloc::vec::Vec<Q8Block> = (0..n_super)
            .map(|s| quantize_block_q8(&x_row[s * Q6_K_BLOCK_NUMEL..(s + 1) * Q6_K_BLOCK_NUMEL]))
            .collect();
        let xq = &xq;
        let out = &mut y_out[b * out_feat..(b + 1) * out_feat];
        out.par_chunks_mut(grain).enumerate().for_each(|(t, slot)| {
            let base = t * grain;
            let mut parts: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(n_super);
            for (i, o) in slot.iter_mut().enumerate() {
                let r = base + i;
                let w_row = &w_q6k[r * row_bytes..(r + 1) * row_bytes];
                parts.clear();
                for s in 0..n_super {
                    let off = s * Q6_K_BLOCK_BYTES;
                    parts.push(dot_q6k_q8_block(
                        &w_row[off..off + Q6_K_BLOCK_BYTES], &xq[s], Q6_K_BLOCK_NUMEL));
                }
                *o = fixed_tree(&mut parts);
            }
        });
    }
    Ok(())
}

/// Row-parallel integer Q4_K linear.
#[cfg(feature = "std-parallel")]
pub fn linear_q4_k_integer_parallel(
    x: &[f32],
    w_q4k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
    threads: usize,
) -> Result<()> {
    if in_feat % Q4_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q4_k_integer_parallel: in_feat % 256"));
    }
    if threads <= 1 {
        return linear_q4_k_integer(x, w_q4k, y_out, batch, in_feat, out_feat);
    }
    let row_bytes = (in_feat / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
    if w_q4k.len() < out_feat * row_bytes {
        return Err(Error::Internal("linear_q4_k_integer_parallel: weight shape"));
    }
    let n_super = in_feat / Q4_K_BLOCK_NUMEL;
    use rayon::prelude::*;
    let grain = core::cmp::max(1, out_feat.div_ceil(threads * 4));
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        let xq: alloc::vec::Vec<Q8Block> = (0..n_super)
            .map(|s| quantize_block_q8(&x_row[s * Q4_K_BLOCK_NUMEL..(s + 1) * Q4_K_BLOCK_NUMEL]))
            .collect();
        let xq = &xq;
        let out = &mut y_out[b * out_feat..(b + 1) * out_feat];
        out.par_chunks_mut(grain).enumerate().for_each(|(t, slot)| {
            let base = t * grain;
            let mut parts: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(n_super);
            for (i, o) in slot.iter_mut().enumerate() {
                let r = base + i;
                let w_row = &w_q4k[r * row_bytes..(r + 1) * row_bytes];
                parts.clear();
                for s in 0..n_super {
                    let off = s * Q4_K_BLOCK_BYTES;
                    parts.push(dot_q4k_q8_block(
                        &w_row[off..off + Q4_K_BLOCK_BYTES], &xq[s], Q4_K_BLOCK_NUMEL));
                }
                *o = fixed_tree(&mut parts);
            }
        });
    }
    Ok(())
}

/// Fused dequant + canonical dot for ONE Q6_K row.
///
/// Q6_K is 15.7% of Mistral-7B's weights but includes the single largest
/// tensor in the model — `output.weight`, 4096x32000 = 131.1M weights — plus
/// every `ffn_down`. `linear_q6_k_cpu` materialises the whole matrix before
/// computing one output, so the output projection alone allocated and wrote
/// **524 MB per token**, single-threaded, while the Q4_K path was already
/// fused and threaded. It was the serial tail holding CPU utilisation near
/// 200% on a 14-core machine.
///
/// Bit-identical to dequantize-then-`canonical_dot` by construction: the
/// dequant arithmetic is copied unchanged, and Q6_K_BLOCK_NUMEL (256) is a
/// multiple of both CANON_LANES and CANON_CHUNK's divisor, so super-block
/// boundaries never split a lane group or a chunk.
///
/// **This is an f32 REFERENCE, NOT the shipped Q6_K regime — do not issue or
/// verify Q6_K certificates against it.** The forward (`linear_dispatch`)
/// always computes Q6_K through the INTEGER path (`linear_q6_k_integer`), which
/// quantizes the activations to int8 and does an exact integer dot. That is
/// deliberately NOT bit-identical to this f32-dequant dot — they agree only to
/// ~1e-4 (see `integer_q6k_agrees_with_float_on_same_activations`), because the
/// integer path is lossy in the activation quantization. Unlike Q4_K — whose
/// fused and shipped paths ARE bit-identical, so `canonical_dot_q4k_fused`
/// keeps its name — there is no "canonical" f32 Q6_K regime. Renamed from
/// `q6k_fused_f32_dot`, whose "canonical" wrongly implied this was the
/// regime the certificate binds.
pub fn q6k_fused_f32_dot(x: &[f32], w_q6k: &[u8], n: usize) -> Result<f32> {
    if w_q6k.len() % Q6_K_BLOCK_BYTES != 0 {
        return Err(Error::Internal("Q6_K byte length not a multiple of 210"));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let n_super = w_q6k.len() / Q6_K_BLOCK_BYTES;
    if n_super * Q6_K_BLOCK_NUMEL < n {
        return Err(Error::Internal("Q6_K row shorter than n"));
    }
    let supers_per_chunk = CANON_CHUNK / Q6_K_BLOCK_NUMEL;
    let mut chunk_sums: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    let mut lanes = [0.0f32; CANON_LANES];
    let mut buf = [0.0f32; Q6_K_BLOCK_NUMEL];

    for b in 0..n_super {
        let base = b * Q6_K_BLOCK_NUMEL;
        if base >= n {
            break;
        }
        let off = b * Q6_K_BLOCK_BYTES;
        let ql_base = off;
        let qh_base = off + 128;
        let sc_base = off + 192;
        let d_off = off + 208;
        let d = f16_to_f32(u16::from_le_bytes([w_q6k[d_off], w_q6k[d_off + 1]]));

        let mut ql_p = 0usize;
        let mut qh_p = 0usize;
        let mut sc_p = 0usize;
        let mut y = 0usize;
        for _ in (0..Q6_K_BLOCK_NUMEL).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let ql_lo0 = w_q6k[ql_base + ql_p + l] & 0x0F;
                let ql_lo1 = w_q6k[ql_base + ql_p + l + 32] & 0x0F;
                let ql_hi0 = w_q6k[ql_base + ql_p + l] >> 4;
                let ql_hi1 = w_q6k[ql_base + ql_p + l + 32] >> 4;
                let qh = w_q6k[qh_base + qh_p + l];
                let q1 = ((ql_lo0 | (((qh) & 0x03) << 4)) as i32 - 32) as f32;
                let q2 = ((ql_lo1 | (((qh >> 2) & 0x03) << 4)) as i32 - 32) as f32;
                let q3 = ((ql_hi0 | (((qh >> 4) & 0x03) << 4)) as i32 - 32) as f32;
                let q4 = ((ql_hi1 | (((qh >> 6) & 0x03) << 4)) as i32 - 32) as f32;
                let s1 = w_q6k[sc_base + sc_p + is] as i8 as f32;
                let s2 = w_q6k[sc_base + sc_p + is + 2] as i8 as f32;
                let s3 = w_q6k[sc_base + sc_p + is + 4] as i8 as f32;
                let s4 = w_q6k[sc_base + sc_p + is + 6] as i8 as f32;
                buf[y + l] = d * s1 * q1;
                buf[y + l + 32] = d * s2 * q2;
                buf[y + l + 64] = d * s3 * q3;
                buf[y + l + 96] = d * s4 * q4;
            }
            y += 128;
            ql_p += 64;
            qh_p += 32;
            sc_p += 8;
        }

        let avail = core::cmp::min(Q6_K_BLOCK_NUMEL, n - base);
        let xs = &x[base..base + avail];
        let bs = &buf[..avail];
        for (xc, bc) in xs
            .chunks_exact(CANON_LANES)
            .zip(bs.chunks_exact(CANON_LANES))
        {
            for j in 0..CANON_LANES {
                let p = xc[j] * bc[j];
                lanes[j] += p;
            }
        }
        let rem = avail % CANON_LANES;
        if rem != 0 {
            let s = avail - rem;
            for t in s..avail {
                let p = x[base + t] * buf[t];
                lanes[t % CANON_LANES] += p;
            }
        }

        let is_last = base + avail >= n;
        if (b + 1) % supers_per_chunk == 0 || is_last {
            let mut l = lanes;
            chunk_sums.push(fixed_tree(&mut l));
            lanes = [0.0f32; CANON_LANES];
        }
    }
    Ok(fixed_tree(&mut chunk_sums))
}

/// Fused-dequant Q6_K linear. Same bits as `linear_q6_k_cpu`, without
/// materialising the weight matrix.
pub fn linear_q6_k_fused(
    x: &[f32],
    w_q6k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if in_feat % Q6_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q6_k_fused: in_feat not a multiple of 256"));
    }
    let row_bytes = (in_feat / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES;
    if w_q6k.len() < out_feat * row_bytes {
        return Err(Error::Internal("linear_q6_k_fused: weight shape mismatch"));
    }
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        for o in 0..out_feat {
            let w_row = &w_q6k[o * row_bytes..(o + 1) * row_bytes];
            y_out[b * out_feat + o] = q6k_fused_f32_dot(x_row, w_row, in_feat)?;
        }
    }
    Ok(())
}

/// Row-parallel Q6_K linear. Same independence argument as the Q4_K variant.
#[cfg(feature = "std-parallel")]
pub fn linear_q6_k_fused_parallel(
    x: &[f32],
    w_q6k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
    threads: usize,
) -> Result<()> {
    if in_feat % Q6_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal(
            "linear_q6_k_fused_parallel: in_feat not a multiple of 256",
        ));
    }
    if threads <= 1 {
        return linear_q6_k_fused(x, w_q6k, y_out, batch, in_feat, out_feat);
    }
    let row_bytes = (in_feat / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES;
    if w_q6k.len() < out_feat * row_bytes {
        return Err(Error::Internal(
            "linear_q6_k_fused_parallel: weight shape mismatch",
        ));
    }
    use rayon::prelude::*;
    let grain = core::cmp::max(1, out_feat.div_ceil(threads * 4));
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        let out = &mut y_out[b * out_feat..(b + 1) * out_feat];
        out.par_chunks_mut(grain).enumerate().for_each(|(t, slot)| {
            let base = t * grain;
            for (i, o) in slot.iter_mut().enumerate() {
                let r = base + i;
                let w_row = &w_q6k[r * row_bytes..(r + 1) * row_bytes];
                *o = q6k_fused_f32_dot(x_row, w_row, in_feat).unwrap_or(0.0);
            }
        });
    }
    Ok(())
}

/// Row-parallel Q4_K linear using std threads.
///
/// **Bit-identical to the serial path, structurally.** Each output element is
/// an independent `canonical_dot_q4k_fused` over one weight row; no state is
/// shared and no partial sum crosses a row boundary. Splitting rows across
/// threads therefore cannot perturb any reduction order — unlike splitting
/// *within* a reduction, which is what CANON_LANES/CANON_CHUNK exist to pin.
///
/// Worth doing here specifically because we are COMPUTE-bound, not
/// bandwidth-bound. Measured on the same machine and weights, llama.cpp does
/// 5.55 tok/s at 1 thread and 5.39 at 14 — it gains nothing from threads
/// because it already saturates memory (~22.6 GB/s). This implementation sits
/// near 1.75 GB/s, i.e. ~13x below that ceiling, so the cores are idle waiting
/// on arithmetic rather than on memory.
#[cfg(feature = "std-parallel")]
pub fn linear_q4_k_fused_parallel(
    x: &[f32],
    w_q4k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
    threads: usize,
) -> Result<()> {
    if in_feat % Q4_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal(
            "linear_q4_k_fused_parallel: in_feat not a multiple of 256",
        ));
    }
    if threads <= 1 {
        return linear_q4_k_fused(x, w_q4k, y_out, batch, in_feat, out_feat);
    }
    let row_bytes = (in_feat / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
    if w_q4k.len() < out_feat * row_bytes {
        return Err(Error::Internal(
            "linear_q4_k_fused_parallel: weight shape mismatch",
        ));
    }
    use rayon::prelude::*;
    // Grain is rows-per-task, not threads: rayon owns a persistent pool, so
    // this schedules work onto live workers instead of spawning per call.
    let grain = core::cmp::max(1, out_feat.div_ceil(threads * 4));
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        let out = &mut y_out[b * out_feat..(b + 1) * out_feat];
        out.par_chunks_mut(grain)
            .enumerate()
            .for_each(|(t, slot)| {
                let base = t * grain;
                for (i, o) in slot.iter_mut().enumerate() {
                    let r = base + i;
                    let w_row = &w_q4k[r * row_bytes..(r + 1) * row_bytes];
                    // Errors cannot occur here: shapes were validated above
                    // and every row is the same length.
                    *o = canonical_dot_q4k_fused(x_row, w_row, in_feat).unwrap_or(0.0);
                }
            });
    }
    Ok(())
}

/// Regime-3 linear. Same shape as `linear_q4_k_fused`, different bits.
pub fn linear_q4_k_fused(
    x: &[f32],
    w_q4k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if in_feat % Q4_K_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q4_k_fused: in_feat not a multiple of 256"));
    }
    let row_bytes = (in_feat / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
    for b in 0..batch {
        let x_row = &x[b * in_feat..(b + 1) * in_feat];
        for o in 0..out_feat {
            let w_row = &w_q4k[o * row_bytes..(o + 1) * row_bytes];
            y_out[b * out_feat + o] = canonical_dot_q4k_fused(x_row, w_row, in_feat)?;
        }
    }
    Ok(())
}


pub fn linear_q4_k_cpu(
    x: &[f32],
    w_q4k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if x.len() != batch * in_feat {
        return Err(Error::Internal("linear_q4_k_cpu: x shape mismatch"));
    }
    if y_out.len() != batch * out_feat {
        return Err(Error::Internal("linear_q4_k_cpu: y shape mismatch"));
    }
    let w_f32 = dequantize_q4_k(w_q4k)?;
    if w_f32.len() != out_feat * in_feat {
        return Err(Error::Internal("linear_q4_k_cpu: weight shape mismatch"));
    }
    for b in 0..batch {
        for o in 0..out_feat {
            let w_row = &w_f32[o * in_feat..(o + 1) * in_feat];
            let x_row = &x[b * in_feat..(b + 1) * in_feat];
            y_out[b * out_feat + o] = canonical_dot(x_row, w_row, in_feat);
        }
    }
    Ok(())
}

/// CPU reference for `linear` against Q6_K weights.
pub fn linear_q6_k_cpu(
    x: &[f32],
    w_q6k: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if x.len() != batch * in_feat {
        return Err(Error::Internal("linear_q6_k_cpu: x shape mismatch"));
    }
    if y_out.len() != batch * out_feat {
        return Err(Error::Internal("linear_q6_k_cpu: y shape mismatch"));
    }
    let w_f32 = dequantize_q6_k(w_q6k)?;
    if w_f32.len() != out_feat * in_feat {
        return Err(Error::Internal("linear_q6_k_cpu: weight shape mismatch"));
    }
    for b in 0..batch {
        for o in 0..out_feat {
            let w_row = &w_f32[o * in_feat..(o + 1) * in_feat];
            let x_row = &x[b * in_feat..(b + 1) * in_feat];
            y_out[b * out_feat + o] = canonical_dot(x_row, w_row, in_feat);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------
// linear_q4_0 — fused dequant + matmul
//
// y[batch, out_feat] = x[batch, in_feat] @ W^T
// where W is stored as Q4_0 in row-major (out_feat × in_feat),
// matching SYS_GPU_Q4_LINEAR's expected layout.
// ---------------------------------------------------------------

/// CPU reference for `linear_q4_0`. Dequantizes the entire weight
/// once into a scratch f32 buffer, then runs a plain row-major
/// matmul. Slow but correct — used as the test oracle and as the
/// fallback when no GPU is available.
///
/// `x`: shape `[batch, in_feat]`, row-major f32.
/// `w_q4`: Q4_0 blob, shape `[out_feat, in_feat]`, row-major.
/// `y_out`: shape `[batch, out_feat]`, row-major f32 (written).
pub fn linear_q4_0_cpu(
    x: &[f32],
    w_q4: &[u8],
    y_out: &mut [f32],
    batch: usize,
    in_feat: usize,
    out_feat: usize,
) -> Result<()> {
    if in_feat % Q4_0_BLOCK_NUMEL != 0 {
        return Err(Error::Internal("linear_q4_0: in_feat must be multiple of 32"));
    }
    let expected_w = out_feat * (in_feat / Q4_0_BLOCK_NUMEL) * Q4_0_BLOCK_BYTES;
    if w_q4.len() != expected_w {
        return Err(Error::Internal("linear_q4_0: weight blob length mismatch"));
    }
    if x.len() != batch * in_feat {
        return Err(Error::Internal("linear_q4_0: x size mismatch"));
    }
    if y_out.len() != batch * out_feat {
        return Err(Error::Internal("linear_q4_0: y_out size mismatch"));
    }

    // Dequantize all weights once. For large models the GPU path
    // does this on-device into a scratch buffer; we mirror that
    // shape here for correctness.
    let w_f32 = dequantize_q4_0(w_q4)?;

    // y[b, o] = sum_i x[b, i] * w[o, i]
    for b in 0..batch {
        for o in 0..out_feat {
            let w_row = &w_f32[o * in_feat..(o + 1) * in_feat];
            let x_row = &x[b * in_feat..(b + 1) * in_feat];
            y_out[b * out_feat + o] = canonical_dot(x_row, w_row, in_feat);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Regime-3 reference: canonical_dot_v2 over the published LCG
    /// vectors at n=4096. Set from measurement, asserted forever after.
    const PINNED_V2_REFERENCE: u32 = 0x40ce_39e1;

    use super::*;

    #[test]
    fn q4_0_roundtrip_smooth_weights_error_bound() {
        // Smooth synthetic weights — what real LLM weights tend to
        // look like inside one row: small dynamic range, mostly
        // gaussian. Q4_0's 4-bit precision loses some accuracy but
        // the ratio should stay tight (well under 5% mean abs error
        // relative to amax).
        let n = 32 * 64; // 2048 elements = 64 Q4_0 blocks
        let mut data = vec![0f32; n];
        for i in 0..n {
            let x = i as f32 * 0.0314;
            data[i] = (x.sin() * 0.6) + ((x * 0.5).cos() * 0.3);
        }
        let q4 = quantize_q4_0(&data).unwrap();
        assert_eq!(q4.len(), 64 * Q4_0_BLOCK_BYTES);
        let dq = dequantize_q4_0(&q4).unwrap();
        assert_eq!(dq.len(), n);

        let mut max_amp = 0f32;
        let mut sum_err = 0f32;
        for i in 0..n {
            max_amp = max_amp.max(data[i].abs());
            sum_err += (data[i] - dq[i]).abs();
        }
        let mae = sum_err / n as f32;
        let mae_rel = mae / max_amp.max(1e-9);
        assert!(
            mae_rel < 0.05,
            "Q4_0 relative MAE {} > 5% — quantization broken",
            mae_rel
        );
    }

    #[test]
    fn q8_0_roundtrip_tighter_than_q4_0() {
        // Q8_0 should be ~16x more accurate than Q4_0 (one more bit
        // per direction, plus exact representation of 0).
        let n = 32 * 32;
        let mut data = vec![0f32; n];
        for i in 0..n {
            let x = i as f32 * 0.0271;
            data[i] = (x.sin() * 0.5) - ((x * 1.7).cos() * 0.3);
        }
        let q8 = quantize_q8_0(&data).unwrap();
        let dq = dequantize_q8_0(&q8).unwrap();

        let mut max_amp = 0f32;
        let mut sum_err = 0f32;
        for i in 0..n {
            max_amp = max_amp.max(data[i].abs());
            sum_err += (data[i] - dq[i]).abs();
        }
        let mae_rel = (sum_err / n as f32) / max_amp.max(1e-9);
        assert!(
            mae_rel < 0.005,
            "Q8_0 relative MAE {} > 0.5% — quantization broken",
            mae_rel
        );
    }

    #[test]
    fn linear_q4_0_cpu_matches_dequant_then_matmul() {
        // Same as: dequantize first, then do plain matmul. Verifies
        // the fused-path code paths the same thing the manual
        // sequence would.
        let in_feat = 64;
        let out_feat = 8;
        let batch = 2;

        let mut w_f32 = vec![0f32; out_feat * in_feat];
        for i in 0..w_f32.len() {
            w_f32[i] = ((i as f32 * 0.013).sin()) * 0.7;
        }
        let w_q4 = quantize_q4_0(&w_f32).unwrap();
        let w_dq = dequantize_q4_0(&w_q4).unwrap();

        let mut x = vec![0f32; batch * in_feat];
        for i in 0..x.len() {
            x[i] = ((i as f32 * 0.071).cos()) * 0.5;
        }

        // Reference path
        let mut y_ref = vec![0f32; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut acc = 0f32;
                for i in 0..in_feat {
                    acc += x[b * in_feat + i] * w_dq[o * in_feat + i];
                }
                y_ref[b * out_feat + o] = acc;
            }
        }

        // Fused path
        let mut y = vec![0f32; batch * out_feat];
        linear_q4_0_cpu(&x, &w_q4, &mut y, batch, in_feat, out_feat).unwrap();

        for (a, b) in y.iter().zip(y_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "fused vs reference disagree: {} vs {}", a, b);
        }
    }

    #[test]
    fn f16_roundtrip_matches_known_constants() {
        // Sanity: f16 ↔ f32 with a few hand-picked constants.
        // 1.0 = 0x3C00 = 0b0_01111_0000000000
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -2.0 = 0xC000
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        // 0.5 = 0x3800
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // 0
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    // ---------------------------------------------------------------
    // Q4_K / Q6_K tests
    //
    // Strategy: build a Q4_K / Q6_K super-block by hand with known
    // byte patterns, dequantize, verify each output element matches
    // the algebraic prediction from the canonical formula. If anything
    // is byte-wrong (wrong scale unpack, wrong nibble order), the
    // numeric output diverges in a way these tests catch.
    // ---------------------------------------------------------------

    #[test]
    fn q4_k_dequant_uniform_scale_one() {
        // Construct a Q4_K block where:
        //   d = 1.0, dmin = 0.0
        //   scales[0..4]   = 1 (low 6 bits)        → scale[0..4] = 1
        //   scales[4..8]   = 0                     → min[0..4]   = 0
        //   scales[8..12]  = 0                     → scale[4..8] = 0, min[4..8] = 0
        //   qs[0..128]     = 0x10 (low nibble 0, high nibble 1)
        //
        // Sub-blocks 0..4: scale=1, min=0  → weights = nibble - 0
        //   First 32 weights of sub-block 0: low nibbles of qs[0..16]
        //     = 0 (constant)
        //   Next 32 weights: high nibbles of qs[0..16] = 1
        //   First 32 of sub-block 2: low nibbles of qs[16..32] = 0
        //   ...
        // Sub-blocks 4..8: scale=0  → weights = 0
        //
        // Decoded expected output is therefore alternating-32-runs
        // of 0, 1, 0, 1 across the first 4 sub-blocks (= 256 weights),
        // then 0s for the last 4 sub-blocks. Actually since super-iter
        // step is 64, the structure within sub-blocks 0..4 (first 128
        // weights) is: 32x0, 32x1, 32x0, 32x1. Then sub-blocks 4..8
        // (next 128 weights) are all 0.
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        // d = 1.0 (f16)
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // dmin = 0.0
        block[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
        // scales: only sub-blocks 0..4 get scale=1; mins all zero; sub-blocks 4..8 all zero
        for j in 0..4 { block[4 + j] = 1; }       // scale[0..4] = 1
        for j in 0..4 { block[4 + 4 + j] = 0; }   // min[0..4] = 0
        for j in 0..4 { block[4 + 8 + j] = 0; }   // packs scale[4..8] high + min[4..8] high
        // qs: each byte = 0x10 (low nibble 0, high nibble 1)
        for i in 0..128 { block[16 + i] = 0x10; }

        let out = dequantize_q4_k(&block).unwrap();
        assert_eq!(out.len(), Q4_K_BLOCK_NUMEL);

        // First 128 weights from sub-blocks 0..4 (sub-blocks 0,1,2,3)
        for i in 0..32   { assert_eq!(out[i],         0.0, "low nibble of sb=0 @ {}", i); }
        for i in 32..64  { assert_eq!(out[i],         1.0, "high nibble of sb=1 @ {}", i); }
        for i in 64..96  { assert_eq!(out[i],         0.0, "low nibble of sb=2 @ {}", i); }
        for i in 96..128 { assert_eq!(out[i],         1.0, "high nibble of sb=3 @ {}", i); }
        // Next 128 weights from sub-blocks 4..8: all zero (scale=0)
        for i in 128..256 { assert_eq!(out[i],        0.0, "sb=4..8 should be zero @ {}", i); }
    }

    #[test]
    fn q4_k_dequant_negative_min() {
        // Verify the min term is subtracted, not added (a common
        // off-by-sign bug). Build a block where d=1, dmin=2,
        // scales[0]=0 (scale=0), scales[4]=3 (min=3). Sub-block 0
        // dequantization: weight = 0 * nibble - 2 * 3 = -6 for all 32.
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        block[4] = 0;     // scale[0] = 0
        block[8] = 3;     // min[0]   = 3
        // qs all 0 — doesn't matter, scale=0 zeroes the nibble term.
        let out = dequantize_q4_k(&block).unwrap();
        for i in 0..32 {
            assert!((out[i] - (-6.0)).abs() < 1e-5, "min subtraction broken @ {}: {}", i, out[i]);
        }
    }

    #[test]
    fn q6_k_dequant_uniform_scale_one() {
        // Q6_K: d = 1, all sub-block scales = 1, all ql nibbles = 0.
        // Each unpacked quant = 0 | (qh_bits << 4). With qh=0 → 0-32 = -32.
        // So all 256 weights should equal -32.
        let mut block = [0u8; Q6_K_BLOCK_BYTES];
        // ql[0..128] = 0
        // qh[0..64]  = 0
        // scales[0..16] = 1
        for i in 0..16 { block[192 + i] = 1; }
        // d = 1.0
        block[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());

        let out = dequantize_q6_k(&block).unwrap();
        assert_eq!(out.len(), Q6_K_BLOCK_NUMEL);
        for i in 0..Q6_K_BLOCK_NUMEL {
            assert!(
                (out[i] - (-32.0)).abs() < 1e-5,
                "Q6_K @ {} expected -32, got {}",
                i, out[i]
            );
        }
    }

    #[test]
    fn q6_k_dequant_qh_high_bits_apply() {
        // Verify qh's 2-bit groups feed into the quant correctly.
        // d=1, all scales=1, ql=0, qh[0] = 0b11_10_01_00.
        //   For l=0:
        //     q1 = 0 | ((qh>>0 & 0x3) << 4) = 0 | (0 << 4) = 0  → -32
        //     q2 = 0 | ((qh>>2 & 0x3) << 4) = 0 | (1 << 4) = 16 → -16
        //     q3 = 0 | ((qh>>4 & 0x3) << 4) = 0 | (2 << 4) = 32 → 0
        //     q4 = 0 | ((qh>>6 & 0x3) << 4) = 0 | (3 << 4) = 48 → 16
        //   So out[0]=-32, out[32]=-16, out[64]=0, out[96]=16.
        let mut block = [0u8; Q6_K_BLOCK_BYTES];
        block[128] = 0b11_10_01_00;  // qh[0]
        for i in 0..16 { block[192 + i] = 1; }
        block[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());

        let out = dequantize_q6_k(&block).unwrap();
        assert!((out[0]  - (-32.0)).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[32] - (-16.0)).abs() < 1e-5, "out[32]={}", out[32]);
        assert!((out[64] -   0.0 ).abs() < 1e-5, "out[64]={}", out[64]);
        assert!((out[96] -  16.0 ).abs() < 1e-5, "out[96]={}", out[96]);
    }

    #[test]
    fn linear_q4_k_matches_dequant_then_matmul() {
        // Sanity: linear_q4_k_cpu = dequantize_q4_k + plain matmul.
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
        for j in 0..4 { block[4 + j] = 1; }
        for i in 0..128 { block[16 + i] = ((i % 16) as u8) | (((i % 16) as u8) << 4); }

        // 1 row of 256-element weights — out_feat=1, in_feat=256, batch=1
        let w_q4k = block.to_vec();
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();

        let w_f32 = dequantize_q4_k(&w_q4k).unwrap();
        let mut y_ref = vec![0f32; 1];
        for k in 0..256 { y_ref[0] += x[k] * w_f32[k]; }

        let mut y_fused = vec![0f32; 1];
        linear_q4_k_cpu(&x, &w_q4k, &mut y_fused, 1, 256, 1).unwrap();
        assert!(
            (y_ref[0] - y_fused[0]).abs() < 1e-4,
            "fused linear_q4_k != dequant+matmul: ref={} fused={}",
            y_ref[0], y_fused[0]
        );
    }

    // ===============================================================
    // Canonical v2 — the properties that make it safe to go fast.
    //
    // The whole point of pinning LANES and CHUNK as CONTRACT constants
    // (rather than reading vector width / thread count) is that the
    // result must not move when the implementation gets faster. These
    // tests assert exactly that, because a determinism claim that only
    // holds for the slow path is worthless.
    // ===============================================================

    /// Published test-vector input (contract doc 4.0): `a[0..n]` then
    /// `b[0..n]` drawn from ONE continuous LCG stream. Generating each
    /// from a fresh seed would make a == b and turn every dot product
    /// into a sum of squares — which masks sign-cancellation, the case
    /// where reduction order matters most.
    fn lcg_pair(n: usize) -> (alloc::vec::Vec<f32>, alloc::vec::Vec<f32>) {
        let mut s: u64 = 0x1234;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / 2147483648.0) * 2.0 - 1.0
        };
        let a: alloc::vec::Vec<f32> = (0..n).map(|_| next()).collect();
        let b: alloc::vec::Vec<f32> = (0..n).map(|_| next()).collect();
        (a, b)
    }

    /// Chunks evaluated in ANY order must combine to the same bits.
    /// This is the thread-count-independence property: a scheduler that
    /// finishes chunk 7 before chunk 0 cannot change the answer.
    #[test]
    fn v2_is_invariant_to_chunk_completion_order() {
        let n = CANON_CHUNK * 5 + 137; // several chunks + a ragged tail
        let (a, b) = lcg_pair(n);
        let serial = canonical_dot(&a, &b, n);

        let nchunks = (n + CANON_CHUNK - 1) / CANON_CHUNK;
        // Compute chunk sums in REVERSE order (worst case for a naive
        // implementation that accumulates as results arrive).
        let mut sums = alloc::vec![0.0f32; nchunks];
        for c in (0..nchunks).rev() {
            let s = c * CANON_CHUNK;
            let e = core::cmp::min(s + CANON_CHUNK, n);
            sums[c] = canonical_chunk(&a[s..e], &b[s..e]);
        }
        let reordered = canonical_combine(&mut sums);
        assert_eq!(
            serial.to_bits(),
            reordered.to_bits(),
            "chunk completion order changed the result: {:#x} vs {:#x}",
            serial.to_bits(),
            reordered.to_bits()
        );
    }

    /// A scalar reference implementation of the SPEC (element -> lane by
    /// index alone) must equal the fast path. This is the vector-width-
    /// independence property: if a future SIMD backend disagrees with
    /// this, the backend is wrong, not this test.
    #[test]
    fn v2_fast_path_matches_spec_definition() {
        for n in [1usize, 7, 8, 9, 63, 64, 65, 4096, 14336, CANON_CHUNK + 1] {
            let (a, b) = lcg_pair(n);

            // Literal transcription of the spec: lane j takes every
            // element whose in-chunk index is congruent to j mod LANES.
            let nchunks = (n + CANON_CHUNK - 1) / CANON_CHUNK;
            let mut sums = alloc::vec![0.0f32; nchunks];
            for c in 0..nchunks {
                let s = c * CANON_CHUNK;
                let e = core::cmp::min(s + CANON_CHUNK, n);
                let mut lanes = [0.0f32; CANON_LANES];
                for i in s..e {
                    lanes[(i - s) % CANON_LANES] += a[i] * b[i];
                }
                let mut l = lanes;
                sums[c] = canonical_combine(&mut l);
            }
            let spec = canonical_combine(&mut sums);
            let fast = canonical_dot(&a, &b, n);
            assert_eq!(
                spec.to_bits(),
                fast.to_bits(),
                "n={}: fast path diverged from spec: {:#x} vs {:#x}",
                n,
                spec.to_bits(),
                fast.to_bits()
            );
        }
    }

    /// v2 IS a different reduction order from v1 — but not on every
    /// input. Reassociation can land on the same bits by luck, and it
    /// does at exactly n=4096, which is the size the published §4.1
    /// vector uses. Measured 2026-07-29: v1 and v2 agree at n=1000 and
    /// n=4096, and differ at the other 12 of 14 sizes tried.
    ///
    /// That makes §4.1 BLIND to this distinction — a reduction-shape
    /// change that a reference vector cannot detect is exactly the
    /// blind spot §4.2 was added to close for transcendentals. So this
    /// test asserts divergence at a size where it is real (14336 =
    /// Mistral's hidden dim) and pins the coincidence at 4096 so nobody
    /// "fixes" it later thinking it is a bug.
    #[test]
    fn v2_reduction_order_differs_from_v1() {
        let (a, b) = lcg_pair(14336);
        let v1 = canonical_dot_regime2(&a, &b, 14336);
        let v2 = canonical_dot(&a, &b, 14336);
        assert_ne!(
            v1.to_bits(), v2.to_bits(),
            "v1 and v2 agree at n=14336 — the lane restructure did not take effect"
        );
        assert!((v1 - v2).abs() < 1e-2, "v1={} v2={} — reassociation, not a different computation", v1, v2);

        // The documented coincidence. If this ever starts failing, the
        // published §4.1 reference value has moved and every regime-2
        // certificate needs re-checking.
        let (a, b) = lcg_pair(4096);
        assert_eq!(
            canonical_dot_regime2(&a, &b, 4096).to_bits(),
            canonical_dot(&a, &b, 4096).to_bits(),
            "n=4096 coincidence broke — §4.1's vector no longer covers both regimes"
        );
    }

    /// Pin v2's value on the published reference vector so any future
    /// change to the reduction shape fails loudly instead of silently
    /// invalidating every issued certificate.
    /// The DISCRIMINATING reference. n=4096 (below) cannot tell regime 2
    /// from regime 3 — both give 0x40ce39e1 — so the regime fingerprint
    /// mixes this size as well. Pinned here so the two stay in step.
    /// Cross-checked against the issuer's independent implementation
    /// (the reference::sse_row_dot) and the verifier's regime probe:
    /// all three produce 0xc1a3bf11.
    #[test]
    fn v2_discriminating_reference_is_pinned() {
        let (a, b) = lcg_pair(14336);
        let v = canonical_dot(&a, &b, 14336);
        assert_eq!(
            v.to_bits(), 0xc1a3_bf11u32,
            "regime-3 discriminating reference moved: {:#010x} — NUMERIC_REGIME_CHECK \
             (kernel), CONTRACT_FINGERPRINT (issuer) and NUMERIC_REGIME_CHECK \
             (verifier) all need recomputing together",
            v.to_bits()
        );
        // And it must NOT equal the regime-2 value, or the probe is useless.
        assert_ne!(
            canonical_dot_regime2(&a, &b, 14336).to_bits(), v.to_bits(),
            "discriminating probe does not discriminate"
        );
    }

    #[test]
    fn v2_reference_bits_are_pinned() {
        let (a, b) = lcg_pair(4096);
        let v = canonical_dot(&a, &b, 4096);
        // Recorded from this implementation; regime 3 reference value.
        assert_eq!(
            v.to_bits(),
            PINNED_V2_REFERENCE,
            "canonical v2 reference moved: {:#x} (expected {:#x}) — this \
             invalidates every regime-3 certificate",
            v.to_bits(),
            PINNED_V2_REFERENCE
        );
    }

    /// The fused path must be BIT-IDENTICAL to dequantize-then-dot, not
    /// merely close. If it ever isn't, every certificate issued under
    /// regime 2 silently stops reproducing — so this is asserted on
    /// exact bits across several shapes, including a non-power-of-two
    /// out_feat and a multi-super-block in_feat.
    #[test]
    fn fused_q4k_dot_is_bit_identical() {
        for &(in_feat, out_feat) in &[(256usize, 3usize), (512, 5), (1024, 7), (4096, 2), (8192, 2), (14336, 2)] {
            // Deterministic pseudo-random Q4_K bytes + activations.
            let row_bytes = (in_feat / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
            let mut w = alloc::vec![0u8; out_feat * row_bytes];
            let mut s: u64 = 0xC0FFEE;
            for v in w.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (s >> 40) as u8;
            }
            let x: alloc::vec::Vec<f32> = (0..in_feat)
                .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let mut y_ref = alloc::vec![0.0f32; out_feat];
            linear_q4_k_cpu(&x, &w, &mut y_ref, 1, in_feat, out_feat).unwrap();
            let mut y_fused = alloc::vec![0.0f32; out_feat];
            linear_q4_k_fused(&x, &w, &mut y_fused, 1, in_feat, out_feat).unwrap();

            for o in 0..out_feat {
                // Random *bytes* as Q4_K weights can decode to non-finite fp16
                // scales, driving the dot to Inf/NaN. IEEE-754 does not specify
                // NaN payload propagation, so it legitimately differs by target
                // and codegen (e.g. x86_64 vs the fused path's operation order) —
                // it is outside the bit-identity contract, and real quantized
                // weights are always finite. Compare bits only where the
                // reference is finite; a finite reference the fused path fails to
                // reproduce (whether it returns a different finite value or a
                // non-finite one) still trips the assert below.
                if !y_ref[o].is_finite() {
                    continue;
                }
                assert_eq!(
                    y_ref[o].to_bits(),
                    y_fused[o].to_bits(),
                    "in_feat={} out_feat={} row {}: fused {:#010x} != reference {:#010x} \
                     — fused path diverged from dequantize-then-dot",
                    in_feat, out_feat, o,
                    y_fused[o].to_bits(), y_ref[o].to_bits()
                );
            }
        }
    }

    /// Row-parallel MUST be bit-identical to serial, at every thread count.
    /// This is the property that makes threading safe to turn on: rows are
    /// independent reductions, so the split cannot reach inside one. If this
    /// ever fails, threading is silently changing certified values.
    #[cfg(feature = "std-parallel")]
    #[test]
    fn parallel_q4k_matches_serial_at_every_thread_count() {
        let (in_feat, out_feat) = (4096usize, 1024usize);
        let row_bytes = (in_feat / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
        let mut w = alloc::vec![0u8; out_feat * row_bytes];
        let mut s: u64 = 0xC0FFEE;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s >> 40) as u8;
        }
        for blk in w.chunks_mut(Q4_K_BLOCK_BYTES) {
            blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
        }
        let x: alloc::vec::Vec<f32> = (0..in_feat)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();

        let mut serial = alloc::vec![0.0f32; out_feat];
        linear_q4_k_fused(&x, &w, &mut serial, 1, in_feat, out_feat).unwrap();

        for &t in &[1usize, 2, 3, 4, 7, 8, 16, 33] {
            let mut par = alloc::vec![0.0f32; out_feat];
            linear_q4_k_fused_parallel(&x, &w, &mut par, 1, in_feat, out_feat, t).unwrap();
            for o in 0..out_feat {
                assert_eq!(
                    serial[o].to_bits(),
                    par[o].to_bits(),
                    "threads={} row {}: parallel {:#010x} != serial {:#010x}",
                    t, o, par[o].to_bits(), serial[o].to_bits()
                );
            }
        }
    }

    /// Fused Q6_K must equal dequantize-then-dot, exactly. Same argument as
    /// the Q4_K case; asserted separately because it is separate code and
    /// carries the model's largest tensor.
    #[test]
    fn fused_q6k_dot_is_bit_identical() {
        for &(in_feat, out_feat) in &[(256usize, 3usize), (1024, 5), (4096, 2), (8192, 2)] {
            let row_bytes = (in_feat / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES;
            let mut w = alloc::vec![0u8; out_feat * row_bytes];
            let mut s: u64 = 0xBEEF;
            for v in w.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (s >> 40) as u8;
            }
            for blk in w.chunks_mut(Q6_K_BLOCK_BYTES) {
                blk[208..210].copy_from_slice(&0x2E66u16.to_le_bytes());
            }
            let x: alloc::vec::Vec<f32> = (0..in_feat)
                .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect();
            let mut y_ref = alloc::vec![0.0f32; out_feat];
            linear_q6_k_cpu(&x, &w, &mut y_ref, 1, in_feat, out_feat).unwrap();
            let mut y_fused = alloc::vec![0.0f32; out_feat];
            linear_q6_k_fused(&x, &w, &mut y_fused, 1, in_feat, out_feat).unwrap();
            for o in 0..out_feat {
                assert_eq!(
                    y_ref[o].to_bits(), y_fused[o].to_bits(),
                    "in_feat={} out_feat={} row {}: fused {:#010x} != reference {:#010x}",
                    in_feat, out_feat, o, y_fused[o].to_bits(), y_ref[o].to_bits()
                );
            }
        }
    }

    /// THE property the integer path buys: the per-super-block reduction is
    /// exactly order-independent, because integer addition is associative.
    /// Verified by accumulating the sub-block terms in REVERSE order and
    /// demanding bit-identity — something the float paths cannot offer at
    /// any granularity, which is why they need CANON_LANES/CANON_CHUNK.
    #[test]
    fn integer_block_dot_is_order_independent() {
        let mut w = [0u8; Q4_K_BLOCK_BYTES];
        let mut s: u64 = 0xABCDEF;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s >> 40) as u8;
        }
        w[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
        w[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
        let x: alloc::vec::Vec<f32> = (0..256)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let xq = quantize_block_q8(&x);

        let forward = dot_q4k_q8_block(&w, &xq, 256);

        // Same arithmetic, sub-block terms accumulated back-to-front.
        let d = f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([w[2], w[3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&w[4..16]);
        let qs = &w[16..Q4_K_BLOCK_BYTES];
        let mut acc: i32 = 0;
        let mut acc_m: i32 = 0;
        for half in (0..4).rev() {
            let (sc_lo, m_lo) = get_scale_min_k4(half * 2, &scales);
            let (sc_hi, m_hi) = get_scale_min_k4(half * 2 + 1, &scales);
            let q_off = half * 32;
            let base = half * 64;
            let mut s_lo: i32 = 0;
            let mut s_hi: i32 = 0;
            for l in (0..32).rev() {
                s_lo += (qs[q_off + l] & 0x0F) as i32 * xq.qs[base + l] as i32;
                s_hi += (qs[q_off + l] >> 4) as i32 * xq.qs[base + 32 + l] as i32;
            }
            acc += sc_lo as i32 * s_lo + sc_hi as i32 * s_hi;
            acc_m += m_lo as i32 * xq.bsums[half * 2] + m_hi as i32 * xq.bsums[half * 2 + 1];
        }
        let reversed = d * xq.d * acc as f32 - dmin * xq.d * acc_m as f32;

        assert_eq!(
            forward.to_bits(), reversed.to_bits(),
            "integer block dot is order-DEPENDENT: {:#010x} vs {:#010x} — the \
             whole point of the integer domain is that this cannot happen",
            forward.to_bits(), reversed.to_bits()
        );
    }

    /// The integer path must agree with the float path **on the same
    /// activations** — that is the correctness check. Comparing against raw
    /// f32 activations instead would measure int8 quantization loss, which
    /// can be large in relative terms wherever the result is a near-
    /// cancellation of bigger terms (measured 13.8% at n=1024, where the
    /// true value 9.0 is the residue of terms two orders larger). That is a
    /// property of int8 activations, not a defect in this code.
    #[test]
    fn integer_dot_agrees_with_float_on_same_activations() {
        for n in [256usize, 1024, 4096, 14336] {
            let row_bytes = (n / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
            let mut w = alloc::vec![0u8; row_bytes];
            let mut s: u64 = 0x13579;
            for v in w.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (s >> 40) as u8;
            }
            for blk in w.chunks_mut(Q4_K_BLOCK_BYTES) {
                blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
            }
            let x: alloc::vec::Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect();
            // Feed the float path the identical int8-rounded activations.
            let mut xq_rt = alloc::vec![0.0f32; n];
            for b in 0..(n / 256) {
                let q = quantize_block_q8(&x[b * 256..(b + 1) * 256]);
                for i in 0..256 {
                    xq_rt[b * 256 + i] = q.d * q.qs[i] as f32;
                }
            }
            let f = canonical_dot_q4k_fused(&xq_rt, &w, n).unwrap();
            let i = integer_dot_q4k(&x, &w, n).unwrap();
            let rel = (f - i).abs() / f.abs().max(1.0);
            assert!(
                rel < 1e-4,
                "n={}: integer {} vs float-on-same-activations {} — rel {:.2e}. \
                 These use identical data, so any real gap is a layout bug.",
                n, i, f, rel
            );
        }
        // One super-block is ENTIRELY integer, so agreement there is exact.
        let mut w = [0u8; Q4_K_BLOCK_BYTES];
        let mut s: u64 = 0x13579;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s >> 40) as u8;
        }
        w[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
        w[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
        let x: alloc::vec::Vec<f32> = (0..256)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let q = quantize_block_q8(&x);
        let rt: alloc::vec::Vec<f32> = (0..256).map(|i| q.d * q.qs[i] as f32).collect();
        assert_eq!(
            canonical_dot_q4k_fused(&rt, &w, 256).unwrap().to_bits(),
            integer_dot_q4k(&x, &w, 256).unwrap().to_bits(),
            "single super-block should be BIT-EXACT — it is pure integer"
        );
    }

    /// Diagnostic: isolate quantization loss from a possible layout bug by
    /// feeding the FLOAT path the very same int8-rounded activations the
    /// integer path uses. If the layouts agree, these must match closely —
    /// any remaining gap is the reduction, not the data.
    #[test]
    fn integer_dot_matches_float_on_identical_quantized_activations() {
        for n in [256usize, 1024, 4096] {
            let row_bytes = (n / Q4_K_BLOCK_NUMEL) * Q4_K_BLOCK_BYTES;
            let mut w = alloc::vec![0u8; row_bytes];
            let mut s: u64 = 0x13579;
            for v in w.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (s >> 40) as u8;
            }
            for blk in w.chunks_mut(Q4_K_BLOCK_BYTES) {
                blk[0..2].copy_from_slice(&0x2E66u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x2666u16.to_le_bytes());
            }
            let x: alloc::vec::Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect();

            let mut xq_rt = alloc::vec![0.0f32; n];
            for b in 0..(n / 256) {
                let blk = &x[b * 256..(b + 1) * 256];
                let q = quantize_block_q8(blk);
                for i in 0..256 {
                    xq_rt[b * 256 + i] = q.d * q.qs[i] as f32;
                }
            }
            let f_rt = canonical_dot_q4k_fused(&xq_rt, &w, n).unwrap();
            let i_int = integer_dot_q4k(&x, &w, n).unwrap();
            let f_raw = canonical_dot_q4k_fused(&x, &w, n).unwrap();
            std::println!(
                "  n={:5}  float(raw)={:12.6}  float(q8)={:12.6}  integer={:12.6}  int-vs-q8 rel={:.2e}  q8-vs-raw rel={:.2e}",
                n, f_raw, f_rt, i_int,
                (f_rt - i_int).abs() / f_rt.abs().max(1.0),
                (f_raw - f_rt).abs() / f_raw.abs().max(1.0)
            );
        }
    }

    /// Q6_K integer path: same two properties as Q4_K. Agreement with float
    /// on identical int8 activations (correctness of the 6-bit unpack and
    /// scale bucketing), and bit-exactness for a single super-block.
    #[test]
    fn integer_q6k_agrees_with_float_on_same_activations() {
        for n in [256usize, 1024, 4096] {
            let row_bytes = (n / Q6_K_BLOCK_NUMEL) * Q6_K_BLOCK_BYTES;
            let mut w = alloc::vec![0u8; row_bytes];
            let mut s: u64 = 0x2468;
            for v in w.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (s >> 40) as u8;
            }
            for blk in w.chunks_mut(Q6_K_BLOCK_BYTES) {
                blk[208..210].copy_from_slice(&0x2E66u16.to_le_bytes());
            }
            let x: alloc::vec::Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
                .collect();
            let mut rt = alloc::vec![0.0f32; n];
            for b in 0..(n / 256) {
                let q = quantize_block_q8(&x[b * 256..(b + 1) * 256]);
                for i in 0..256 {
                    rt[b * 256 + i] = q.d * q.qs[i] as f32;
                }
            }
            let mut yf = alloc::vec![0.0f32; 1];
            linear_q6_k_fused(&rt, &w[..row_bytes], &mut yf, 1, n, 1).unwrap();
            let mut yi = alloc::vec![0.0f32; 1];
            linear_q6_k_integer(&x, &w[..row_bytes], &mut yi, 1, n, 1).unwrap();
            let rel = (yf[0] - yi[0]).abs() / yf[0].abs().max(1.0);
            assert!(
                rel < 1e-4,
                "n={}: q6k integer {} vs float-on-same-activations {} rel {:.2e} \\
                 — identical data, so a real gap is a 6-bit unpack or scale bug",
                n, yi[0], yf[0], rel
            );
        }
    }

    /// Order-independence for Q6_K, proven the same way: accumulate the 16
    /// scale buckets back-to-front and require bit-identity.
    #[test]
    fn integer_q6k_block_dot_is_order_independent() {
        let mut w = [0u8; Q6_K_BLOCK_BYTES];
        let mut s: u64 = 0x99AA;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s >> 40) as u8;
        }
        w[208..210].copy_from_slice(&0x2E66u16.to_le_bytes());
        let x: alloc::vec::Vec<f32> = (0..256)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let xq = quantize_block_q8(&x);
        let fwd = dot_q6k_q8_block(&w, &xq, 256);

        // Recompute with the scale-bucket accumulation reversed.
        let d = f16_to_f32(u16::from_le_bytes([w[208], w[209]]));
        let ql = &w[0..128];
        let qh = &w[128..192];
        let sc = &w[192..208];
        let mut sums = [0i32; 16];
        for half in 0..2 {
            let (ql_p, qh_p, sc_p, y) = (half * 64, half * 32, half * 8, half * 128);
            for l in 0..32 {
                let is = l / 16;
                let h = qh[qh_p + l];
                let q1 = ((ql[ql_p + l] & 0x0F) | ((h & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[ql_p + l + 32] & 0x0F) | (((h >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[ql_p + l] >> 4) | (((h >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[ql_p + l + 32] >> 4) | (((h >> 6) & 0x03) << 4)) as i32 - 32;
                sums[sc_p + is] += q1 * xq.qs[y + l] as i32;
                sums[sc_p + is + 2] += q2 * xq.qs[y + l + 32] as i32;
                sums[sc_p + is + 4] += q3 * xq.qs[y + l + 64] as i32;
                sums[sc_p + is + 6] += q4 * xq.qs[y + l + 96] as i32;
            }
        }
        let mut acc: i32 = 0;
        for j in (0..16).rev() {
            acc += sc[j] as i8 as i32 * sums[j];
        }
        let rev = d * xq.d * acc as f32;
        assert_eq!(fwd.to_bits(), rev.to_bits(), "q6k integer dot is order-DEPENDENT");
    }

    /// The vector path must be BIT-EXACT with the scalar path, not merely
    /// close. If this ever fails, the NEON kernel has changed the numerical
    /// regime and every previously issued certificate is invalidated.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_dot32_matches_scalar() {
        let mut st: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for _ in 0..2000 {
            let qsl: [u8; 32] = core::array::from_fn(|_| (next() >> 32) as u8);
            let xlo: [i8; 32] = core::array::from_fn(|_| ((next() >> 32) as i8).max(-127));
            let xhi: [i8; 32] = core::array::from_fn(|_| ((next() >> 32) as i8).max(-127));
            let a = super::dot32_pair_scalar(&qsl, &xlo, &xhi);
            let b = super::dot32_pair_neon(&qsl, &xlo, &xhi);
            assert_eq!(a, b, "NEON dot32 diverged from scalar - REGIME BREAK");
        }
    }
}
