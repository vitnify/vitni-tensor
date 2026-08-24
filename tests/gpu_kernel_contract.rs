//! Portable guard: the GPU reference kernels' ALGORITHM matches the CPU
//! canonical reduction — checkable in CI with no GPU present.
//!
//! `kernels/canonical_matmul.metal` and `kernels/canonical_matmul.cu` are
//! verified bit-for-bit against this crate on real hardware (Apple M3 Max via
//! `kernels/conformance-metal`, NVIDIA T4 via `kernels/cuda_conformance.cu` —
//! see `kernels/RESULTS-nvidia-t4.txt`). But those runs need a GPU. This test
//! is their CPU-side twin: it re-implements, INDEPENDENTLY, the exact reduction
//! the kernels perform (8 lanes by `i % 8`, a fixed pairwise tree, and the
//! 8192-element chunk split) and asserts it equals `canonical_dot_pub`
//! bit-for-bit, plus reproduces the shipped matmul pin.
//!
//! Why a hand-copied replica instead of calling `canonical_dot` directly: the
//! Metal/CUDA kernels are frozen source that cannot auto-update. If someone
//! changes `canonical_dot`, this replica (which mirrors the kernels) stops
//! matching and the test fails LOUDLY — the signal that the GPU kernels must be
//! re-synced, before a divergent regime ships.

use vitni_tensor::ops::quant::canonical_dot_pub;

const LANES: usize = 8; // CANON_LANES
const CHUNK: usize = 8192; // CANON_CHUNK

/// Mirror of the kernels' `fixed_tree`: hlen = (len+1)/2, part[t] =
/// part[2t] + part[2t+1] (or the lone tail), in place.
fn k_fixed_tree(part: &mut [f32]) -> f32 {
    let mut len = part.len();
    if len == 0 {
        return 0.0;
    }
    while len > 1 {
        let hlen = (len + 1) / 2;
        for t in 0..hlen {
            let u = 2 * t;
            part[t] = if u + 1 < len { part[u] + part[u + 1] } else { part[u] };
        }
        len = hlen;
    }
    part[0]
}

/// Mirror of the kernels' `canonical_chunk`.
fn k_chunk(x: &[f32], w: &[f32]) -> f32 {
    let n = x.len();
    let mut lanes = [0.0f32; LANES];
    let full = n - (n % LANES);
    let mut i = 0;
    while i < full {
        for j in 0..LANES {
            let p = x[i + j] * w[i + j];
            lanes[j] += p;
        }
        i += LANES;
    }
    while i < n {
        let j = i % LANES;
        let p = x[i] * w[i];
        lanes[j] += p;
        i += 1;
    }
    k_fixed_tree(&mut lanes)
}

/// Mirror of the kernels' `canonical_matmul` per-element reduction.
fn k_dot(x: &[f32], w: &[f32], n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    if n <= CHUNK {
        return k_chunk(&x[..n], &w[..n]);
    }
    let nchunks = (n + CHUNK - 1) / CHUNK;
    let mut sums = vec![0.0f32; nchunks];
    for c in 0..nchunks {
        let s = c * CHUNK;
        let e = core::cmp::min(s + CHUNK, n);
        sums[c] = k_chunk(&x[s..e], &w[s..e]);
    }
    k_fixed_tree(&mut sums)
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

#[test]
fn gpu_kernel_reduction_matches_cpu_contract_bit_for_bit() {
    // Lengths span: tails (non-multiples of 8), the exact chunk size, a
    // one-past spill into two chunks, Mistral's FFN K, and a many-chunk case.
    for &n in &[1usize, 7, 8, 9, 63, 64, 333, 4096, 8192, 8193, 14336, 40000] {
        let x = lcg_vec(n, 0xA11CE ^ (n as u64).wrapping_mul(0x9E3779B1));
        let w = lcg_vec(n, 0xB0B ^ (n as u64).wrapping_mul(0x85EBCA77));
        let expected = canonical_dot_pub(&x, &w, n);
        let got = k_dot(&x, &w, n);
        assert_eq!(
            expected.to_bits(),
            got.to_bits(),
            "GPU kernel algorithm diverged from canonical_dot at n={} \
             (cpu {:#010x} vs kernel {:#010x}) — re-sync kernels/canonical_matmul.*",
            n,
            expected.to_bits(),
            got.to_bits()
        );
    }
}

#[test]
fn gpu_kernel_algorithm_reproduces_matmul_pin() {
    // Exact (4,64,4) LCG vector + FNV-1a from matmul.rs::matmul_reduction_bits_are_pinned.
    const PINNED_MATMUL_HASH: u64 = 0x8a42_8433_686d_13af;
    let (m, k, n) = (4usize, 64usize, 4usize);
    let mut s: u64 = 0x1234;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();

    // matmul via the kernel algorithm: out[i,j] = k_dot(row_i, col_j).
    let mut out = vec![0.0f32; m * n];
    let mut col = vec![0.0f32; k];
    for j in 0..n {
        for kk in 0..k {
            col[kk] = b[kk * n + j];
        }
        for i in 0..m {
            out[i * n + j] = k_dot(&a[i * k..i * k + k], &col, k);
        }
    }

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in &out {
        for byte in x.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    assert_eq!(
        h, PINNED_MATMUL_HASH,
        "GPU kernel algorithm no longer reproduces the certificate pin"
    );
}
