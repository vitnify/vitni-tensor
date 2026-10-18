// q4k_imma_test.cu — Q4_K linear on int8 tensor cores (WMMA), bit-identical to the
// integer reference. This is the production-kernel core of Tier-2 (regime-3).
//
// out[M,N] = Σ_s fixed_tree_s( d_s·dx_s·acc_s - dmin_s·dx_s·accm_s ), where per super-
// block s (256 elems = 8 sub-blocks of 32): acc_s = Σ_j sc_j·(Σ_l nib·xq), an EXACT
// integer sum. The Σ_l nib·xq sub-block dots run on int8 tensor cores (16x16x16 WMMA);
// the per-sub-block integer scale sc_j and the per-super-block float + fixed-tree
// combine happen in the epilogue. Integer accumulation is associative, so the tensor
// core's internal order does not change a bit.
//
// Build: nvcc -O2 -arch=sm_75 q4k_imma_test.cu -o q4k_imma   (int8 tensor cores: sm_75+)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// ---- shared helpers (host + device) ----
__host__ __device__ static inline float f16f(unsigned short h) {
    unsigned s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0u) { if (m == 0u) bits = s << 31; else { unsigned mm = m; int ee = 1; while ((mm & 0x400u) == 0u) { mm <<= 1; ee -= 1; } mm &= 0x3FFu; bits = (s << 31) | ((unsigned)(127 - 15 + ee) << 23) | (mm << 13); } }
    else if (e == 0x1Fu) bits = (s << 31) | 0x7F800000u | (m << 13);
    else bits = (s << 31) | ((e + 127u - 15u) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
__host__ __device__ static inline void gsm(unsigned j, const unsigned char* sc, unsigned* s, unsigned* m) {
    if (j < 4u) { *s = sc[j] & 0x3Fu; *m = sc[j + 4u] & 0x3Fu; }
    else { *s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6) << 4); *m = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4); }
}
__host__ __device__ static inline float ftree(float* p, unsigned n) {
    if (n == 0u) return 0.0f;
    while (n > 1u) { unsigned h = (n + 1u) / 2u; for (unsigned t = 0u; t < h; t++) { unsigned u = 2u * t; p[t] = (u + 1u < n) ? (p[u] + p[u + 1u]) : p[u]; } n = h; }
    return p[0];
}

// ---- CPU reference (serial integer, == engine's dot_q4k_q8_block + canonical combine) ----
static void q8_quant(const float* x, float* dx, signed char* qs, int* bs, int nsuper) {
    for (int s = 0; s < nsuper; s++) {
        const float* xb = x + s * 256; float amax = 0; for (int i = 0; i < 256; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        float d = amax / 127.0f, inv = d != 0 ? 1.0f / d : 0.0f; dx[s] = d;
        signed char* qo = qs + s * 256; int* b = bs + s * 8; for (int g = 0; g < 8; g++) b[g] = 0;
        for (int i = 0; i < 256; i++) { float sc = xb[i] * inv; float bias = sc >= 0 ? 0.5f : -0.5f; int q = (int)(sc + bias); q = q < -127 ? -127 : (q > 127 ? 127 : q); qo[i] = (signed char)q; b[i / 32] += q; }
    }
}
static float ref_dot(const unsigned char* wrow, const float* dx, const signed char* xq, const int* bs, int nsuper) {
    float parts[256];
    for (int s = 0; s < nsuper; s++) {
        const unsigned char* blk = wrow + s * 144; float d = f16f(blk[0] | (blk[1] << 8)), dmin = f16f(blk[2] | (blk[3] << 8));
        const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
        const signed char* x = xq + s * 256; const int* b = bs + s * 8;
        int acc = 0, accm = 0;
        for (int half = 0; half < 4; half++) {
            unsigned s1, m1, s2, m2; gsm(half * 2, sc, &s1, &m1); gsm(half * 2 + 1, sc, &s2, &m2);
            int qoff = half * 32, base = half * 64, slo = 0, shi = 0;
            for (int l = 0; l < 32; l++) { slo += (int)(qs[qoff + l] & 0x0F) * (int)x[base + l]; shi += (int)(qs[qoff + l] >> 4) * (int)x[base + 32 + l]; }
            acc += (int)s1 * slo + (int)s2 * shi; accm += (int)m1 * b[half * 2] + (int)m2 * b[half * 2 + 1];
        }
        float a1 = d * dx[s], a2 = dmin * dx[s]; parts[s] = a1 * (float)acc - a2 * (float)accm;
    }
    return ftree(parts, nsuper);
}

// ---- WMMA kernel: one warp per 16x16 output tile ----
__global__ void q4k_imma(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                         float* out, unsigned M, unsigned N, unsigned nsuper) {
    unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned tilesN = N / 16u, tm = warp / tilesN, tn = warp % tilesN;
    if (tm * 16u >= M) return;
    extern __shared__ unsigned char smem[];
    signed char* na = (signed char*)smem;            // [16][32] weight nibbles
    signed char* nbuf = na + 16 * 32;                // [16][32] token xq ([n][k])
    int* sjt = (int*)(nbuf + 16 * 32);               // [16][16] sub-block dot tile
    int* acc = sjt + 256;                            // [16][16]
    int* accm = acc + 256;                           // [16][16]
    unsigned* scrow = (unsigned*)(accm + 256);       // [16] per-row sub-block scale
    unsigned* mrow = scrow + 16;                     // [16] per-row sub-block min
    float* parts = (float*)(mrow + 16);              // [16][16][nsuper]

    for (unsigned i = lane; i < 256u; i += 32u) { acc[i] = 0; accm[i] = 0; }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    for (unsigned s = 0u; s < nsuper; s++) {
        // reset integer accumulators for this super-block
        for (unsigned i = lane; i < 256u; i += 32u) { acc[i] = 0; accm[i] = 0; }
        __syncwarp();
        for (unsigned sub = 0u; sub < 8u; sub++) {           // 8 sub-blocks (half*2 + lo/hi)
            unsigned half = sub >> 1, hi = sub & 1u;
            unsigned qoff = half * 32u, base = half * 64u + hi * 32u;
            // stage weight nibbles na[r][l] and token xq nbuf[c][l] for this sub-block
            for (unsigned i = lane; i < 16u * 32u; i += 32u) {
                unsigned r = i >> 5, l = i & 31u;
                unsigned wr = tm * 16u + r;
                const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
                unsigned q = qs[qoff + l];
                na[r * 32u + l] = (signed char)(hi ? (q >> 4) : (q & 0x0Fu));
                unsigned tc = tn * 16u + r;                  // reuse r as token index for nbuf
                nbuf[r * 32u + l] = xq[((size_t)tc * nsuper + s) * 256u + base + l];
            }
            // per-row scale & min for this sub-block
            if (lane < 16u) {
                unsigned wr = tm * 16u + lane;
                const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u;
                gsm(sub, scb, &scrow[lane], &mrow[lane]);
            }
            __syncwarp();
            // tensor-core dot: s_j[16][16] = na[16][32] . nbuf[16][32]^T  (two k=16 steps)
            wmma::fill_fragment(fc, 0);
            wmma::load_matrix_sync(fa, na, 32); wmma::load_matrix_sync(fb, nbuf, 32); wmma::mma_sync(fc, fa, fb, fc);
            wmma::load_matrix_sync(fa, na + 16, 32); wmma::load_matrix_sync(fb, nbuf + 16, 32); wmma::mma_sync(fc, fa, fb, fc);
            wmma::store_matrix_sync(sjt, fc, 16, wmma::mem_row_major);
            __syncwarp();
            // epilogue: acc[r][c] += sc[r]·s_j[r][c];  accm[r][c] += m[r]·bsums[c][sub]
            for (unsigned i = lane; i < 256u; i += 32u) {
                unsigned r = i >> 4, c = i & 15u;
                acc[i] += (int)scrow[r] * sjt[i];
                unsigned tc = tn * 16u + c;
                accm[i] += (int)mrow[r] * bsums[((size_t)tc * nsuper + s) * 8u + sub];
            }
            __syncwarp();
        }
        // per-super-block float part[r][c][s]
        for (unsigned i = lane; i < 256u; i += 32u) {
            unsigned r = i >> 4, c = i & 15u;
            unsigned wr = tm * 16u + r, tc = tn * 16u + c;
            const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u;
            float d = f16f(blk[0] | (blk[1] << 8)), dmin = f16f(blk[2] | (blk[3] << 8));
            float dxv = dx[(size_t)tc * nsuper + s];
            float a1 = d * dxv, a2 = dmin * dxv;
            parts[(i * nsuper) + s] = a1 * (float)acc[i] - a2 * (float)accm[i];
        }
        __syncwarp();
    }
    // fixed-tree combine across super-blocks, write out[r][c]
    for (unsigned i = lane; i < 256u; i += 32u) {
        unsigned r = i >> 4, c = i & 15u;
        float tmp[256];
        for (unsigned s = 0u; s < nsuper; s++) tmp[s] = parts[i * nsuper + s];
        float v = ftree(tmp, nsuper);
        unsigned wr = tm * 16u + r, tc = tn * 16u + c;
        if (wr < M && tc < N) out[(size_t)wr * N + tc] = v;
    }
}

// ---- Two-pass: pass 1 stages each weight ONCE and reuses it across all N token-tiles
// (weights read once total, not N/16x), writing per-super-block parts to global; pass 2
// does the fixed-tree combine. Bit-identical to the single-pass kernel. NT = N/16 tiles. ----
__global__ void q4k_imma_p1(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                            float* parts, unsigned M, unsigned N, unsigned nsuper) {
    unsigned tm = blockIdx.x; if (tm * 16u >= M) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned NT = N / 16u;
    extern __shared__ unsigned char smem[];
    signed char* na = (signed char*)smem;                 // [16][32] weights (staged once/sub-block)
    signed char* nbuf = na + 16 * 32;                     // [16][32] activations (per N-tile)
    int* sjt = (int*)(nbuf + 16 * 32);                    // [16][16]
    int* acc = sjt + 256;                                 // [NT][16][16]
    int* accm = acc + (size_t)NT * 256;                   // [NT][16][16]
    unsigned* scrow = (unsigned*)(accm + (size_t)NT * 256); // [16]
    unsigned* mrow = scrow + 16;                          // [16]

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = lane; i < NT * 256u; i += 32u) { acc[i] = 0; accm[i] = 0; }
        __syncwarp();
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned half = sub >> 1, hi = sub & 1u, qoff = half * 32u, base = half * 64u + hi * 32u;
            for (unsigned i = lane; i < 16u * 32u; i += 32u) {       // stage weights ONCE per (s,sub)
                unsigned r = i >> 5, l = i & 31u, wr = tm * 16u + r;
                const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
                unsigned q = qs[qoff + l];
                na[r * 32u + l] = (signed char)(hi ? (q >> 4) : (q & 0x0Fu));
            }
            if (lane < 16u) { const unsigned char* scb = W + (size_t)(tm * 16u + lane) * nsuper * 144u + s * 144u + 4u; gsm(sub, scb, &scrow[lane], &mrow[lane]); }
            __syncwarp();
            for (unsigned nt = 0u; nt < NT; nt++) {                  // reuse staged weights across all token-tiles
                for (unsigned i = lane; i < 16u * 32u; i += 32u) {
                    unsigned c = i >> 5, l = i & 31u, tc = nt * 16u + c;
                    nbuf[c * 32u + l] = xq[((size_t)tc * nsuper + s) * 256u + base + l];
                }
                __syncwarp();
                wmma::fill_fragment(fc, 0);
                wmma::load_matrix_sync(fa, na, 32); wmma::load_matrix_sync(fb, nbuf, 32); wmma::mma_sync(fc, fa, fb, fc);
                wmma::load_matrix_sync(fa, na + 16, 32); wmma::load_matrix_sync(fb, nbuf + 16, 32); wmma::mma_sync(fc, fa, fb, fc);
                wmma::store_matrix_sync(sjt, fc, 16, wmma::mem_row_major);
                __syncwarp();
                int* a = acc + (size_t)nt * 256; int* am = accm + (size_t)nt * 256;
                for (unsigned i = lane; i < 256u; i += 32u) {
                    unsigned r = i >> 4, c = i & 15u, tc = nt * 16u + c;
                    a[i] += (int)scrow[r] * sjt[i];
                    am[i] += (int)mrow[r] * bsums[((size_t)tc * nsuper + s) * 8u + sub];
                }
                __syncwarp();
            }
        }
        for (unsigned nt = 0u; nt < NT; nt++) {                      // write this super-block's parts
            int* a = acc + (size_t)nt * 256; int* am = accm + (size_t)nt * 256;
            for (unsigned i = lane; i < 256u; i += 32u) {
                unsigned r = i >> 4, c = i & 15u, wr = tm * 16u + r, tc = nt * 16u + c;
                const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u;
                float d = f16f(blk[0] | (blk[1] << 8)), dmin = f16f(blk[2] | (blk[3] << 8)), dxv = dx[(size_t)tc * nsuper + s];
                float a1 = d * dxv, a2 = dmin * dxv;
                parts[((size_t)wr * N + tc) * nsuper + s] = a1 * (float)a[i] - a2 * (float)am[i];
            }
        }
        __syncwarp();
    }
}
__global__ void q4k_combine(const float* parts, float* out, unsigned M, unsigned N, unsigned nsuper) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= M * N) return;
    float tmp[256];
    const float* p = parts + (size_t)idx * nsuper;
    for (unsigned s = 0u; s < nsuper; s++) tmp[s] = p[s];
    out[idx] = ftree(tmp, nsuper);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 256, N = argc > 2 ? atoi(argv[2]) : 64, nsuper = argc > 3 ? atoi(argv[3]) : 20;
    int K = nsuper * 256, iters = argc > 4 ? atoi(argv[4]) : 50;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("== %s (CC %d.%d) ==  Q4_K-IMMA  M=%d N=%d K=%d (nsuper=%d)\n", p.name, p.major, p.minor, M, N, K, nsuper);

    // synthetic Q4_K weights + f32 activations
    size_t wbytes = (size_t)M * nsuper * 144;
    unsigned char* hW = (unsigned char*)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) hW[i] = (unsigned char)((i * 131 + 7) % 256);
    // make the two f16 scales per block sane (not NaN/inf): small positive
    for (int m = 0; m < M; m++) for (int s = 0; s < nsuper; s++) { unsigned char* b = hW + ((size_t)m * nsuper + s) * 144; b[0] = 0x00; b[1] = 0x2C; b[2] = 0x00; b[3] = 0x28; } // d=~0.0625, dmin=~0.03
    float* hX = (float*)malloc((size_t)N * K * 4);
    for (size_t i = 0; i < (size_t)N * K; i++) hX[i] = ((float)((i * 37 + 11) % 1000) / 500.0f - 1.0f);

    // CPU reference
    float* dxh = (float*)malloc((size_t)N * nsuper * 4); signed char* qsh = (signed char*)malloc((size_t)N * nsuper * 256); int* bsh = (int*)malloc((size_t)N * nsuper * 8 * 4);
    for (int n = 0; n < N; n++) q8_quant(hX + (size_t)n * K, dxh + n * nsuper, qsh + (size_t)n * nsuper * 256, bsh + n * nsuper * 8, nsuper);
    float* ref = (float*)malloc((size_t)M * N * 4);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++)
        ref[(size_t)m * N + n] = ref_dot(hW + (size_t)m * nsuper * 144, dxh + (size_t)n * nsuper, qsh + (size_t)n * nsuper * 256, bsh + (size_t)n * nsuper * 8, nsuper);

    // GPU
    unsigned char* dW; float* dX; float* ddx; signed char* dqs; int* dbs; float* dout;
    cudaMalloc(&dW, wbytes); cudaMalloc(&dX, (size_t)N * K * 4); cudaMalloc(&ddx, (size_t)N * nsuper * 4);
    cudaMalloc(&dqs, (size_t)N * nsuper * 256); cudaMalloc(&dbs, (size_t)N * nsuper * 8 * 4); cudaMalloc(&dout, (size_t)M * N * 4);
    cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice); cudaMemcpy(dX, hX, (size_t)N * K * 4, cudaMemcpyHostToDevice);
    // upload the host-quantized activations (quantization itself is not under test here)
    cudaMemcpy(ddx, dxh, (size_t)N * nsuper * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dqs, qsh, (size_t)N * nsuper * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dbs, bsh, (size_t)N * nsuper * 8 * 4, cudaMemcpyHostToDevice);

    unsigned warps = (M / 16) * (N / 16);
    size_t shmem = (16 * 32 + 16 * 32) + (256 + 256 + 256) * 4 + 16 * 4 + 16 * 4 + (size_t)256 * nsuper * 4;
    cudaFuncSetAttribute(q4k_imma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    q4k_imma<<<warps, 32, shmem>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("launch error: %s (shmem=%zu)\n", cudaGetErrorString(e), shmem); return 1; }
    float* gpu = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
    size_t diff = 0, first = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &ref[i], 4); memcpy(&b, &gpu[i], 4); if (a != b) { if (first == (size_t)-1) first = i; diff++; } }
    printf("bit-identity: %zu/%d diffs%s", diff, M * N, diff ? "" : "  -> BIT-IDENTICAL\n");
    if (diff) printf("   first @ %zu  ref=%g gpu=%g\n", first, ref[first], gpu[first]);

    cudaEvent_t s0, e0; cudaEventCreate(&s0); cudaEventCreate(&e0);
    cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_imma<<<warps, 32, shmem>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
    cudaEventRecord(e0); cudaEventSynchronize(e0); float ms = 0; cudaEventElapsedTime(&ms, s0, e0); ms /= iters;
    printf("v1 single-pass : %.3f ms  %.1f int8-TOPS\n", ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12);

    // ---- v2: two-pass, weight-reuse ----
    float* dparts; cudaMalloc(&dparts, (size_t)M * N * nsuper * 4);
    unsigned NT = N / 16u;
    size_t sh2 = (16 * 32 + 16 * 32) + 256 * 4 + (size_t)NT * 256 * 4 * 2 + 16 * 4 + 16 * 4;
    cudaFuncSetAttribute(q4k_imma_p1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2);
    q4k_imma_p1<<<M / 16, 32, sh2>>>(dW, ddx, dqs, dbs, dparts, M, N, nsuper);
    q4k_combine<<<((size_t)M * N + 127) / 128, 128>>>(dparts, dout, M, N, nsuper);
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e2 != cudaSuccess) { printf("v2 launch error: %s (sh2=%zu)\n", cudaGetErrorString(e2), sh2); return 1; }
    float* gpu2 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu2, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
    size_t d2 = 0, f2 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &ref[i], 4); memcpy(&b, &gpu2[i], 4); if (a != b) { if (f2 == (size_t)-1) f2 = i; d2++; } }
    printf("v2 two-pass bit-identity: %zu/%d diffs%s", d2, M * N, d2 ? "" : "  -> BIT-IDENTICAL\n");
    if (d2) printf("   first @ %zu ref=%g gpu=%g\n", f2, ref[f2], gpu2[f2]);
    cudaEventRecord(s0); for (int it = 0; it < iters; it++) { q4k_imma_p1<<<M / 16, 32, sh2>>>(dW, ddx, dqs, dbs, dparts, M, N, nsuper); q4k_combine<<<((size_t)M * N + 127) / 128, 128>>>(dparts, dout, M, N, nsuper); }
    cudaEventRecord(e0); cudaEventSynchronize(e0); float ms2 = 0; cudaEventElapsedTime(&ms2, s0, e0); ms2 /= iters;
    printf("v2 two-pass    : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1)\n", ms2, 2.0 * M * N * K / (ms2 * 1e-3) / 1e12, ms / ms2);
    return 0;
}
