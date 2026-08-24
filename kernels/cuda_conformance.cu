// Self-contained CUDA conformance harness for the canonical reduction contract.
//
// Host code = the CPU reference (fixed_tree / canonical_chunk / canonical_dot /
// matmul), anchored to the shipped pin 0x8a428433686d13af. Device code = two
// kernels: `plain` (bare * and +, so nvcc's --fmad flag controls fusion) and
// `rn` (the shipped __fmul_rn/__fadd_rn kernel, fusion-proof by construction).
//
// Compile it TWO ways and run each:
//   nvcc --fmad=false -Xcompiler -ffp-contract=off -O2  -> "safe"  build
//   nvcc --fmad=true  -Xcompiler -ffp-contract=off -O2  -> "fused" build (control)
//
// Expected:
//   safe : host pin OK, plain MATCH, rn MATCH
//   fused: host pin OK, plain DIVERGES (the gap), rn MATCH (intrinsics win)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CANON_LANES 8u
#define CANON_CHUNK 8192u
static const uint64_t PINNED_MATMUL_HASH = 0x8a428433686d13afULL;

// =============================== CPU reference ===============================
static float h_fixed_tree(std::vector<float>& part) {
    size_t len = part.size();
    if (len == 0) return 0.0f;
    while (len > 1) {
        size_t hlen = (len + 1) / 2;
        for (size_t t = 0; t < hlen; t++) {
            size_t u = 2 * t;
            part[t] = (u + 1 < len) ? (part[u] + part[u + 1]) : part[u];
        }
        len = hlen;
    }
    return part[0];
}
static float h_canonical_chunk(const float* x, const float* w, size_t n) {
    float lanes[CANON_LANES];
    for (unsigned j = 0; j < CANON_LANES; j++) lanes[j] = 0.0f;
    size_t full = n - (n % CANON_LANES);
    size_t i = 0;
    for (; i < full; i += CANON_LANES)
        for (unsigned j = 0; j < CANON_LANES; j++) {
            float p = x[i + j] * w[i + j];
            lanes[j] += p;
        }
    for (; i < n; i++) {
        unsigned j = i % CANON_LANES;
        float p = x[i] * w[i];
        lanes[j] += p;
    }
    std::vector<float> lv(lanes, lanes + CANON_LANES);
    return h_fixed_tree(lv);
}
static float h_canonical_dot(const float* x, const float* w, size_t n) {
    if (n == 0) return 0.0f;
    if (n <= CANON_CHUNK) return h_canonical_chunk(x, w, n);
    size_t nchunks = (n + CANON_CHUNK - 1) / CANON_CHUNK;
    std::vector<float> sums(nchunks);
    for (size_t c = 0; c < nchunks; c++) {
        size_t s = c * CANON_CHUNK, e = std::min(s + CANON_CHUNK, n);
        sums[c] = h_canonical_chunk(x + s, w + s, e - s);
    }
    return h_fixed_tree(sums);
}
static std::vector<float> h_matmul(const std::vector<float>& a, const std::vector<float>& b,
                                   size_t m, size_t k, size_t n) {
    std::vector<float> out(m * n, 0.0f), col(k, 0.0f);
    for (size_t j = 0; j < n; j++) {
        if (k == 0) continue;
        for (size_t kk = 0; kk < k; kk++) col[kk] = b[kk * n + j];
        for (size_t i = 0; i < m; i++)
            out[i * n + j] = h_canonical_dot(&a[i * k], col.data(), k);
    }
    return out;
}

// =============================== device kernels ===============================
__device__ __forceinline__ float d_fixed_tree(float* part, unsigned len) {
    if (len == 0u) return 0.0f;
    while (len > 1u) {
        unsigned hlen = (len + 1u) / 2u;
        for (unsigned t = 0u; t < hlen; t++) {
            unsigned u = 2u * t;
            part[t] = (u + 1u < len) ? (part[u] + part[u + 1u]) : part[u];
        }
        len = hlen;
    }
    return part[0];
}
// PLAIN: bare * and + — nvcc's --fmad flag decides whether these fuse.
__device__ __forceinline__ float d_chunk_plain(const float* a, const float* b,
        unsigned row, unsigned col, unsigned K, unsigned N, unsigned s, unsigned e) {
    float lanes[CANON_LANES];
    for (unsigned j = 0; j < CANON_LANES; j++) lanes[j] = 0.0f;
    unsigned n = e - s, full = n - (n % CANON_LANES), i = 0;
    for (; i < full; i += CANON_LANES)
        for (unsigned j = 0; j < CANON_LANES; j++) {
            unsigned idx = s + i + j;
            float p = a[row * K + idx] * b[idx * N + col];
            lanes[j] += p;
        }
    for (; i < n; i++) {
        unsigned j = i % CANON_LANES, idx = s + i;
        float p = a[row * K + idx] * b[idx * N + col];
        lanes[j] += p;
    }
    return d_fixed_tree(lanes, CANON_LANES);
}
// RN: shipped kernel — round-to-nearest intrinsics, fusion-proof.
__device__ __forceinline__ float d_chunk_rn(const float* a, const float* b,
        unsigned row, unsigned col, unsigned K, unsigned N, unsigned s, unsigned e) {
    float lanes[CANON_LANES];
    for (unsigned j = 0; j < CANON_LANES; j++) lanes[j] = 0.0f;
    unsigned n = e - s, full = n - (n % CANON_LANES), i = 0;
    for (; i < full; i += CANON_LANES)
        for (unsigned j = 0; j < CANON_LANES; j++) {
            unsigned idx = s + i + j;
            float p = __fmul_rn(a[row * K + idx], b[idx * N + col]);
            lanes[j] = __fadd_rn(lanes[j], p);
        }
    for (; i < n; i++) {
        unsigned j = i % CANON_LANES, idx = s + i;
        float p = __fmul_rn(a[row * K + idx], b[idx * N + col]);
        lanes[j] = __fadd_rn(lanes[j], p);
    }
    return d_fixed_tree(lanes, CANON_LANES);
}
template <bool RN>
__global__ void matmul_kernel(const float* a, const float* b, float* out, const unsigned* dims) {
    unsigned M = dims[0], K = dims[1], N = dims[2], total = M * N;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned row = gid / N, col = gid % N;
    if (K == 0u) { out[gid] = 0.0f; return; }
    if (K <= CANON_CHUNK) {
        out[gid] = RN ? d_chunk_rn(a, b, row, col, K, N, 0u, K)
                      : d_chunk_plain(a, b, row, col, K, N, 0u, K);
        return;
    }
    unsigned nchunks = (K + CANON_CHUNK - 1u) / CANON_CHUNK;
    float sums[64];
    for (unsigned c = 0; c < nchunks; c++) {
        unsigned s = c * CANON_CHUNK, e = min(s + CANON_CHUNK, K);
        sums[c] = RN ? d_chunk_rn(a, b, row, col, K, N, s, e)
                     : d_chunk_plain(a, b, row, col, K, N, s, e);
    }
    out[gid] = d_fixed_tree(sums, nchunks);
}

// =============================== helpers ===============================
static uint64_t fnv1a(const std::vector<float>& v) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (float x : v) {
        uint32_t bits; memcpy(&bits, &x, 4);
        for (int b = 0; b < 4; b++) { h ^= (bits >> (8 * b)) & 0xff; h *= 0x100000001b3ULL; }
    }
    return h;
}
static void pin_vector(size_t m, size_t k, size_t n, std::vector<float>& a, std::vector<float>& b) {
    uint64_t s = 0x1234;
    auto rnd = [&]() { s = s * 6364136223846793005ULL + 1; return ((float)(s >> 33) / (float)(1u << 31)) * 2.0f - 1.0f; };
    a.resize(m * k); for (auto& x : a) x = rnd();
    b.resize(k * n); for (auto& x : b) x = rnd();
}
static std::vector<float> rand_vec(size_t len, uint64_t seed) {
    std::vector<float> v(len); uint64_t s = seed;
    for (auto& x : v) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; x = ((float)(s >> 33) / (float)(1u << 31)) * 2.0f - 1.0f; }
    return v;
}
static int64_t ord(float x) { uint32_t b; memcpy(&b, &x, 4); uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u); return (int64_t)o; }
static int64_t ulp(float a, float b) { int64_t d = ord(a) - ord(b); return d < 0 ? -d : d; }

template <bool RN>
static std::vector<float> gpu_matmul(const std::vector<float>& a, const std::vector<float>& b,
                                     unsigned m, unsigned k, unsigned n) {
    float *da, *db, *dout; unsigned* ddims;
    unsigned dims[3] = {m, k, n};
    cudaMalloc(&da, a.size() * 4); cudaMalloc(&db, b.size() * 4);
    cudaMalloc(&dout, (size_t)m * n * 4); cudaMalloc(&ddims, 12);
    cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ddims, dims, 12, cudaMemcpyHostToDevice);
    unsigned total = m * n, tpb = 64, blocks = (total + tpb - 1) / tpb;
    matmul_kernel<RN><<<blocks, tpb>>>(da, db, dout, ddims);
    cudaDeviceSynchronize();
    std::vector<float> out((size_t)m * n);
    cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dout); cudaFree(ddims);
    return out;
}

struct Shape { unsigned m, k, n; const char* label; };

template <bool RN>
static int run_sweep(const char* name) {
    static const Shape shapes[] = {
        {2, 3, 2, "tiny"}, {4, 64, 4, "pin shape"}, {1, 4096, 1, "single dot mid-K"},
        {8, 512, 8, "square-ish"}, {16, 1024, 16, "1K reduction"},
        {2, 8192, 2, "K==CANON_CHUNK"}, {2, 8193, 2, "K+1 spill 2 chunks"},
        {4, 14336, 4, "Mistral FFN K"}, {7, 333, 5, "tail path"},
    };
    int fails = 0;
    printf("  -- %s kernel --\n", name);
    for (auto& sh : shapes) {
        auto a = rand_vec((size_t)sh.m * sh.k, 0xABCD0000ull ^ ((uint64_t)sh.k << 8) ^ sh.m);
        auto b = rand_vec((size_t)sh.k * sh.n, 0x12345678ull ^ ((uint64_t)sh.n << 8) ^ sh.k);
        auto cpu = h_matmul(a, b, sh.m, sh.k, sh.n);
        auto gpu = gpu_matmul<RN>(a, b, sh.m, sh.k, sh.n);
        size_t exact = 0; int64_t mx = 0;
        for (size_t i = 0; i < cpu.size(); i++) {
            uint32_t cb, gb; memcpy(&cb, &cpu[i], 4); memcpy(&gb, &gpu[i], 4);
            if (cb == gb) exact++;
            int64_t u = ulp(cpu[i], gpu[i]); if (u > mx) mx = u;
        }
        bool ok = exact == cpu.size();
        printf("     [%ux%ux%u] %-20s exact %zu/%zu  maxULP %lld  %s\n",
               sh.m, sh.k, sh.n, sh.label, exact, cpu.size(), (long long)mx, ok ? "OK" : "FAIL");
        if (!ok) fails++;
    }
    return fails;
}

int main() {
    cudaDeviceProp prop; int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        printf("no CUDA device\n"); return 2;
    }
    printf("== device ==\n  %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Ground truth: host reference must reproduce the shipped pin.
    std::vector<float> pa, pb; pin_vector(4, 64, 4, pa, pb);
    auto cpu_pin = h_matmul(pa, pb, 4, 64, 4);
    uint64_t cpu_hash = fnv1a(cpu_pin);
    printf("== ground truth ==\n");
    printf("  host FNV over (4,64,4) pin: 0x%016llx\n", (unsigned long long)cpu_hash);
    printf("  PINNED_MATMUL_HASH:         0x%016llx  (%s)\n\n",
           (unsigned long long)PINNED_MATMUL_HASH,
           cpu_hash == PINNED_MATMUL_HASH ? "OK - host IS the contract" : "FAIL - host build fused");
    int fails = (cpu_hash != PINNED_MATMUL_HASH);

    // GPU pin check (both kernels).
    auto gpu_pin_plain = gpu_matmul<false>(pa, pb, 4, 64, 4);
    auto gpu_pin_rn = gpu_matmul<true>(pa, pb, 4, 64, 4);
    printf("== GPU pin (4,64,4) ==\n");
    printf("  plain FNV 0x%016llx  %s\n", (unsigned long long)fnv1a(gpu_pin_plain),
           fnv1a(gpu_pin_plain) == PINNED_MATMUL_HASH ? "== pin" : "!= pin");
    printf("  rn    FNV 0x%016llx  %s\n\n", (unsigned long long)fnv1a(gpu_pin_rn),
           fnv1a(gpu_pin_rn) == PINNED_MATMUL_HASH ? "== pin" : "!= pin");

    printf("== bit-for-bit sweep (GPU vs CPU) ==\n");
    int fp = run_sweep<false>("plain (--fmad flag controls fusion)");
    int fr = run_sweep<true>("rn    (shipped, fusion-proof)");
    fails += fp + fr;

    printf("\nVERDICT: plain %s, rn %s\n",
           fp == 0 ? "MATCH" : "DIVERGES", fr == 0 ? "MATCH" : "DIVERGES");
    return fails == 0 ? 0 : 1;
}
