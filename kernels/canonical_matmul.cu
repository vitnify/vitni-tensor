// ===================================================================
// Deterministic CUDA matmul — the canonical reduction contract on NVIDIA.
//
// This is a faithful port of `canonical_matmul.metal`, which is verified
// bit-for-bit identical to the vitni-tensor CPU reference on Apple Silicon
// (Metal). It is the reference kernel a substrate implements SYS_GPU_MATMUL
// with. It has NOT yet been run on NVIDIA hardware in this tree — see the
// build flags below, which are the whole ballgame for reproducing the CPU
// bits exactly.
//
// REQUIRED nvcc flags (the contract, not tuning):
//   --fmad=false          disable multiply-add fusion (mul and add stay
//                         separately rounded — same as fast-math-off on Metal)
//   -ftz=false            do NOT flush denormals to zero (match CPU IEEE 754)
//   --prec-div=true --prec-sqrt=true   (unused here, but keep IEEE mode)
//   NEVER --use_fast_math (it implies -ftz=true --fmad=true and moves bits)
//
// Belt and suspenders: the products/sums below also use the __fmul_rn /
// __fadd_rn round-to-nearest intrinsics, so even if a caller forgets
// --fmad=false the compiler is not permitted to fuse or reassociate them.
// ===================================================================

#define CANON_LANES 8u
#define CANON_CHUNK 8192u

// Fixed pairwise tree over part[0..len), in place. Exact shape of the CPU
// fixed_tree: hlen = (len+1)/2, part[t] = part[2t] + part[2t+1] (or the lone
// tail element). Ascending-t writes never clobber a not-yet-read input.
__device__ __forceinline__ float fixed_tree(float *part, unsigned int len) {
    if (len == 0u) return 0.0f;
    while (len > 1u) {
        unsigned int hlen = (len + 1u) / 2u;
        for (unsigned int t = 0u; t < hlen; t++) {
            unsigned int u = 2u * t;
            part[t] = (u + 1u < len) ? __fadd_rn(part[u], part[u + 1u]) : part[u];
        }
        len = hlen;
    }
    return part[0];
}

// Reduce ONE contraction chunk [s, e) of a single (row, col) to a scalar.
// a is [M,K] row-major; b is [K,N] row-major; the column is `col`.
__device__ __forceinline__ float canonical_chunk(
    const float *a, const float *b,
    unsigned int row, unsigned int col,
    unsigned int K, unsigned int N,
    unsigned int s, unsigned int e)
{
    float lanes[CANON_LANES];
    for (unsigned int j = 0u; j < CANON_LANES; j++) lanes[j] = 0.0f;

    unsigned int n = e - s;
    unsigned int full = n - (n % CANON_LANES);

    unsigned int i = 0u;
    for (; i < full; i += CANON_LANES) {
        for (unsigned int j = 0u; j < CANON_LANES; j++) {
            unsigned int idx = s + i + j;
            float p = __fmul_rn(a[row * K + idx], b[idx * N + col]); // separate mul
            lanes[j] = __fadd_rn(lanes[j], p);                       // separate add
        }
    }
    for (; i < n; i++) {
        unsigned int j = i % CANON_LANES;
        unsigned int idx = s + i;
        float p = __fmul_rn(a[row * K + idx], b[idx * N + col]);
        lanes[j] = __fadd_rn(lanes[j], p);
    }
    return fixed_tree(lanes, CANON_LANES);
}

// One thread per output element. out is [M,N] row-major. dims = {M, K, N}.
extern "C" __global__ void canonical_matmul(
    const float *a, const float *b, float *out, const unsigned int *dims)
{
    unsigned int M = dims[0], K = dims[1], N = dims[2];
    unsigned int total = M * N;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;                 // guard: grid may exceed M*N
    unsigned int row = gid / N;
    unsigned int col = gid % N;

    if (K == 0u) { out[gid] = 0.0f; return; }

    if (K <= CANON_CHUNK) {
        out[gid] = canonical_chunk(a, b, row, col, K, N, 0u, K);
        return;
    }

    unsigned int nchunks = (K + CANON_CHUNK - 1u) / CANON_CHUNK;
    float sums[64];                           // supports K up to 64*8192 = 512K
    for (unsigned int c = 0u; c < nchunks; c++) {
        unsigned int s = c * CANON_CHUNK;
        unsigned int e = min(s + CANON_CHUNK, K);
        sums[c] = canonical_chunk(a, b, row, col, K, N, s, e);
    }
    out[gid] = fixed_tree(sums, nchunks);
}
