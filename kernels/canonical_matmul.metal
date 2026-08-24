#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Deterministic GPU matmul — the canonical reduction contract on-GPU.
//
// This kernel MUST produce bit-for-bit identical f32 output to the CPU
// reference in `vitni_tensor::ops::quant` (canonical_dot / canonical_chunk
// / fixed_tree), which is the certificate's definition of "the same
// computation". Any divergence breaks cross-device replay.
//
// TWO non-negotiable numerical rules (both enforced here + at compile time):
//   1. ORDER is fixed by the contract, never by the hardware. Element ->
//      lane is `i % CANON_LANES`; lanes and chunks reduce through a fixed
//      pairwise tree in index order. No atomics, no warp shuffles, no
//      hardware-chosen accumulation order.
//   2. FMA CONTRACTION IS FORBIDDEN. The multiply and the add are separate,
//      independently-rounded operations. The host compiles this with
//      fastMathEnabled = false so the compiler cannot fuse `x*w + acc`
//      into a single fma (which rounds once, not twice, and would move the
//      bits). fast-math-off also preserves denormals and honors IEEE 754.
// ===================================================================

constant uint CANON_LANES = 8u;    // contract constant (NOT the SIMD width)
constant uint CANON_CHUNK = 8192u; // contract constant (NOT the thread count)

// Fixed pairwise tree over part[0..len), in place. Exact shape of the CPU
// `fixed_tree`: half = (len+1)/2, part[t] = part[2t] + part[2t+1] (or the
// lone tail element when 2t+1 == len). Ascending-t writes never clobber a
// not-yet-read input, so the in-place update matches the CPU verbatim.
static inline float fixed_tree(thread float *part, uint len) {
    if (len == 0u) return 0.0f;
    while (len > 1u) {
        uint hlen = (len + 1u) / 2u;   // 'half' is a reserved MSL type name
        for (uint t = 0u; t < hlen; t++) {
            uint u = 2u * t;
            part[t] = (u + 1u < len) ? (part[u] + part[u + 1u]) : part[u];
        }
        len = hlen;
    }
    return part[0];
}

// Reduce ONE contraction chunk [s, e) of a single (row, col) to a scalar.
// a is [M,K] row-major; b is [K,N] row-major; the column is `col`.
static inline float canonical_chunk(
    device const float *a, device const float *b,
    uint row, uint col, uint K, uint N, uint s, uint e)
{
    float lanes[CANON_LANES];
    for (uint j = 0u; j < CANON_LANES; j++) lanes[j] = 0.0f;

    uint n = e - s;
    uint full = n - (n % CANON_LANES);

    // Hot loop: CANON_LANES independent chains, element i -> lane (i % LANES).
    uint i = 0u;
    for (; i < full; i += CANON_LANES) {
        for (uint j = 0u; j < CANON_LANES; j++) {
            uint idx = s + i + j;
            float p = a[row * K + idx] * b[idx * N + col]; // separate multiply
            lanes[j] += p;                                 // separate add (no fma)
        }
    }
    // Tail keeps the same element -> lane rule.
    for (; i < n; i++) {
        uint j = i % CANON_LANES;
        uint idx = s + i;
        float p = a[row * K + idx] * b[idx * N + col];
        lanes[j] += p;
    }
    return fixed_tree(lanes, CANON_LANES);
}

// One thread per output element. out is [M,N] row-major.
kernel void canonical_matmul(
    device const float *a    [[buffer(0)]],
    device const float *b    [[buffer(1)]],
    device       float *out  [[buffer(2)]],
    constant     uint  *dims [[buffer(3)]],   // {M, K, N}
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[0], K = dims[1], N = dims[2];
    uint total = M * N;
    if (gid >= total) return;              // guard: grid may exceed M*N
    uint row = gid / N;
    uint col = gid % N;

    if (K == 0u) { out[gid] = 0.0f; return; }

    if (K <= CANON_CHUNK) {
        out[gid] = canonical_chunk(a, b, row, col, K, N, 0u, K);
        return;
    }

    // canonical_dot chunked path: per-chunk canonical_chunk, then a fixed
    // pairwise tree over the chunk sums in INDEX order (not completion order).
    uint nchunks = (K + CANON_CHUNK - 1u) / CANON_CHUNK;
    float sums[64];                        // supports K up to 64*8192 = 512K
    for (uint c = 0u; c < nchunks; c++) {
        uint s = c * CANON_CHUNK;
        uint e = min(s + CANON_CHUNK, K);
        sums[c] = canonical_chunk(a, b, row, col, K, N, s, e);
    }
    out[gid] = fixed_tree(sums, nchunks);
}
