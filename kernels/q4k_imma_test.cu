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
#include <cuda_pipeline.h>
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

// ---- regime-3 combine variant: fixed-order running sum (deterministic, no parts storage) ----
// A register-tiled kernel can't cheaply hold nsuper parts per output for a fixed tree, so
// regime-3 defines the cross-super-block combine as a left-to-right running sum (still a
// fixed order → deterministic/bit-identical). This reference matches the v4 dp4a kernel.
static float ref_dot_rs(const unsigned char* wrow, const float* dx, const signed char* xq, const int* bs, int nsuper) {
    float sum = 0.0f;
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
        float a1 = d * dx[s], a2 = dmin * dx[s]; sum = sum + (a1 * (float)acc - a2 * (float)accm);  // running sum
    }
    return sum;
}

// ---- v4: register-tiled dp4a. Block computes a 64x64 output tile; each thread owns a 4x4
// micro-tile with INTEGER accumulators + the running float sum in REGISTERS (no store_matrix_sync,
// no parts in global). Weights/activations for one super-block are staged (unpacked) in shared and
// reused across the tile. Scales applied in registers. Bit-identical to ref_dot_rs. ----
#define TBM 64
#define TBN 64
__global__ void __launch_bounds__(256) q4k_dp4a_v4(
        const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
        float* out, unsigned M, unsigned N, unsigned nsuper) {
    unsigned tm = blockIdx.x * TBM, tn = blockIdx.y * TBN;
    unsigned tid = threadIdx.x;                       // 0..255
    unsigned trow = tid >> 4, tcol = tid & 15u;       // 16x16 thread grid; each owns 4x4
    __shared__ signed char wsh[TBM][256];             // unpacked weight nibbles (int8)
    __shared__ signed char xsh[TBN][256];             // activation int8
    __shared__ unsigned char scsh[TBM][8], msh[TBM][8];
    __shared__ float dxsh[TBN];
    __shared__ int bsh[TBN][8];

    float outr[4][4]; for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) outr[a][b] = 0.0f;
    float drow[4], dmrow[4];                          // per-owned-row d, dmin (reloaded per super-block)

    for (unsigned s = 0u; s < nsuper; s++) {
        // stage weights (unpack nibbles) : TBM*256 values, 256 threads
        for (unsigned i = tid; i < TBM * 256u; i += 256u) {
            unsigned r = i >> 8, e = i & 255u, wr = tm + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u;
            unsigned q = qs[half * 32u + l];
            wsh[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        // stage activations: TBN*256, and dx/bsums; and per-row scales
        for (unsigned i = tid; i < TBN * 256u; i += 256u) { unsigned c = i >> 8, e = i & 255u, tc = tn + c; xsh[c][e] = xq[((size_t)tc * nsuper + s) * 256u + e]; }
        for (unsigned i = tid; i < TBM * 8u; i += 256u) { unsigned r = i >> 3, sub = i & 7u, wr = tm + r; const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u; unsigned sv, mv; gsm(sub, scb, &sv, &mv); scsh[r][sub] = (unsigned char)sv; msh[r][sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < TBN; i += 256u) { unsigned tc = tn + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[i][g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        // per-owned-row d, dmin
        for (int a = 0; a < 4; a++) { unsigned wr = tm + trow * 4u + a; const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u; drow[a] = f16f(blk[0] | (blk[1] << 8)); dmrow[a] = f16f(blk[2] | (blk[3] << 8)); }
        __syncthreads();
        // compute: each thread owns rows trow*4..+4, cols tcol*4..+4
        int acc[4][4], accm[4][4]; for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { acc[a][b] = 0; accm[a][b] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned half = sub >> 1, hi = sub & 1u, base = half * 64u + hi * 32u;
            for (int a = 0; a < 4; a++) {
                unsigned r = trow * 4u + a; int sc = scsh[r][sub], mm = msh[r][sub];
                for (int b = 0; b < 4; b++) {
                    unsigned c = tcol * 4u + b; int sj = 0;
                    const signed char* wp = &wsh[r][base]; const signed char* xp = &xsh[c][base];
                    for (int k = 0; k < 32; k += 4) {
                        int wv = *(const int*)(wp + k), xv = *(const int*)(xp + k);
                        sj = __dp4a(wv, xv, sj);
                    }
                    acc[a][b] += sc * sj; accm[a][b] += mm * bsh[c][sub];
                }
            }
        }
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            unsigned c = tcol * 4u + b; float dxv = dxsh[c];
            float a1 = drow[a] * dxv, a2 = dmrow[a] * dxv;
            outr[a][b] = outr[a][b] + (a1 * (float)acc[a][b] - a2 * (float)accm[a][b]);   // running sum
        }
        __syncthreads();
    }
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
        unsigned wr = tm + trow * 4u + a, tc = tn + tcol * 4u + b;
        if (wr < M && tc < N) out[(size_t)wr * N + tc] = outr[a][b];
    }
}

// ---- v5: v4's register-tiled dp4a pass-1, but writes per-super-block parts to global so the
// combine is fixed_tree (q4k_combine) — matching the ENGINE's regime-3 linear_q4_k_integer
// (which uses fixed_tree across super-blocks), NOT v4's running sum. This is the on-spec
// production kernel. ----
__global__ void __launch_bounds__(256) q4k_dp4a_v5_p1(
        const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
        float* parts, unsigned M, unsigned N, unsigned nsuper) {
    unsigned tm = blockIdx.x * TBM, tn = blockIdx.y * TBN;
    unsigned tid = threadIdx.x, trow = tid >> 4, tcol = tid & 15u;
    __shared__ signed char wsh[TBM][256];
    __shared__ signed char xsh[TBN][256];
    __shared__ unsigned char scsh[TBM][8], msh[TBM][8];
    __shared__ float dxsh[TBN];
    __shared__ int bsh[TBN][8];
    float drow[4], dmrow[4];
    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = tid; i < TBM * 256u; i += 256u) {
            unsigned r = i >> 8, e = i & 255u, wr = tm + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            wsh[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < TBN * 256u; i += 256u) { unsigned c = i >> 8, e = i & 255u, tc = tn + c; xsh[c][e] = xq[((size_t)tc * nsuper + s) * 256u + e]; }
        for (unsigned i = tid; i < TBM * 8u; i += 256u) { unsigned r = i >> 3, sub = i & 7u, wr = tm + r; const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u; unsigned sv, mv; gsm(sub, scb, &sv, &mv); scsh[r][sub] = (unsigned char)sv; msh[r][sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < TBN; i += 256u) { unsigned tc = tn + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[i][g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        for (int a = 0; a < 4; a++) { unsigned wr = tm + trow * 4u + a; const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u; drow[a] = f16f(blk[0] | (blk[1] << 8)); dmrow[a] = f16f(blk[2] | (blk[3] << 8)); }
        __syncthreads();
        int acc[4][4], accm[4][4]; for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { acc[a][b] = 0; accm[a][b] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned half = sub >> 1, hi = sub & 1u, base = half * 64u + hi * 32u;
            for (int a = 0; a < 4; a++) {
                unsigned r = trow * 4u + a; int sc = scsh[r][sub], mm = msh[r][sub];
                for (int b = 0; b < 4; b++) {
                    unsigned c = tcol * 4u + b; int sj = 0;
                    const signed char* wp = &wsh[r][base]; const signed char* xp = &xsh[c][base];
                    for (int k = 0; k < 32; k += 4) sj = __dp4a(*(const int*)(wp + k), *(const int*)(xp + k), sj);
                    acc[a][b] += sc * sj; accm[a][b] += mm * bsh[c][sub];
                }
            }
        }
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            unsigned wr = tm + trow * 4u + a, tc = tn + tcol * 4u + b, c = tcol * 4u + b; float dxv = dxsh[c];
            float a1 = drow[a] * dxv, a2 = dmrow[a] * dxv;
            if (wr < M && tc < N) parts[((size_t)wr * N + tc) * nsuper + s] = a1 * (float)acc[a][b] - a2 * (float)accm[a][b];
        }
        __syncthreads();
    }
}

// ---- v6: WMMA tensor cores with IN-REGISTER per-sub-block scaling (solves the epilogue bound).
// The per-sub-block integer scale is applied directly to the accumulator fragment — no
// store_matrix_sync, no shared round-trip, no per-sub-block sync. Each thread's fragment
// element->(row,col) is DERIVED at runtime (an mma that yields C[r][c]=r, then =c), so no
// opaque/arch-specific layout is hardcoded. Running-sum combine → matches ref_dot_rs. ----
__global__ void q4k_imma_v6(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                            float* out, unsigned M, unsigned N, unsigned nsuper) {
    unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned tilesN = N / 16u, tm = warp / tilesN, tn = warp % tilesN;
    if (tm * 16u >= M || tn * 16u >= N) return;
    __shared__ signed char na[16][256];     // super-block nibbles (16 weight rows)
    __shared__ signed char xs[16][256];      // super-block xq (16 tokens)
    __shared__ unsigned char scsh[16][8], msh[16][8];
    __shared__ float dxsh[16]; __shared__ int bsh[16][8]; __shared__ float drow[16], dmrow[16];
    __shared__ int probe[16][16];                            // scratch for the layout probe (per warp slot below)

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    // --- derive element->(row,col) for this thread's accumulator lanes, from the hardware itself.
    // The fragment layout is opaque but STABLE per compile, so every mma writes element e to the
    // same (row,col) that store_matrix_sync would. Tag each element with a unique id, store once
    // (row-major), and read back which (row,col) each id landed on. Portable: uses only the
    // documented store path, no hardcoded layout. ---
    int rowmap[8], colmap[8];
    for (int e = 0; e < fc.num_elements; e++) fc.x[e] = (int)(lane * 8u + (unsigned)e);
    wmma::store_matrix_sync(&probe[0][0], fc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = 0; e < 8; e++) { rowmap[e] = 0; colmap[e] = 0; }
    for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) {
        int id = probe[r][c]; unsigned owner = (unsigned)id >> 3;
        if (owner == lane) { int e = id & 7; rowmap[e] = r; colmap[e] = c; }
    }
    __syncwarp();

    float outacc[8]; for (int e = 0; e < 8; e++) outacc[e] = 0.0f;
    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = lane; i < 16u * 256u; i += 32u) {
            unsigned r = i >> 8, e = i & 255u, wr = tm * 16u + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            na[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
            unsigned tc = tn * 16u + r; xs[r][e] = xq[((size_t)tc * nsuper + s) * 256u + e];
        }
        if (lane < 16u) {
            unsigned wr = tm * 16u + lane, tc = tn * 16u + lane;
            const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u;
            for (unsigned sub = 0u; sub < 8u; sub++) { unsigned sv, mv; gsm(sub, scb, &sv, &mv); scsh[lane][sub] = (unsigned char)sv; msh[lane][sub] = (unsigned char)mv; }
            const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u;
            drow[lane] = f16f(blk[0] | (blk[1] << 8)); dmrow[lane] = f16f(blk[2] | (blk[3] << 8));
            dxsh[lane] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[lane][g] = bsums[((size_t)tc * nsuper + s) * 8u + g];
        }
        __syncthreads();
        int acc[8], accm[8]; for (int e = 0; e < 8; e++) { acc[e] = 0; accm[e] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned base = (sub >> 1) * 64u + (sub & 1u) * 32u;
            wmma::fill_fragment(fc, 0);
            wmma::load_matrix_sync(fa, &na[0][base], 256); wmma::load_matrix_sync(fb, &xs[0][base], 256); wmma::mma_sync(fc, fa, fb, fc);
            wmma::load_matrix_sync(fa, &na[0][base + 16], 256); wmma::load_matrix_sync(fb, &xs[0][base + 16], 256); wmma::mma_sync(fc, fa, fb, fc);
            for (int e = 0; e < 8; e++) {                        // in-register scale, no store_matrix_sync
                int r = rowmap[e], cc = colmap[e];
                acc[e] += (int)scsh[r][sub] * fc.x[e];
                accm[e] += (int)msh[r][sub] * bsh[cc][sub];
            }
        }
        for (int e = 0; e < 8; e++) {
            int r = rowmap[e], cc = colmap[e];
            float a1 = drow[r] * dxsh[cc], a2 = dmrow[r] * dxsh[cc];
            outacc[e] += a1 * (float)acc[e] - a2 * (float)accm[e];
        }
        __syncthreads();
    }
    for (int e = 0; e < 8; e++) {
        unsigned wr = tm * 16u + rowmap[e], tc = tn * 16u + colmap[e];
        if (wr < M && tc < N) out[(size_t)wr * N + tc] = outacc[e];   // row-major [M][N], matches ref/v4
    }
}

// ---- v7: v6's in-register-scaled WMMA, now in a cooperative multi-warp block. The block owns ONE
// 16-row M-tile and WN 16-col N-tiles; all WN*32 threads stage the 16 weight rows ONCE per super-
// block (reused across all WN token tiles — the MMQ weight-reuse win), while each warp runs the
// tensor-core mma + in-register per-sub-block scaling for its own token tile. Bit-identical to
// ref_dot_rs; no store_matrix_sync in the hot path. ----
#ifndef WN
#define WN 4
#endif
__global__ void __launch_bounds__(WN * 32) q4k_imma_v7(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                                                       float* out, unsigned M, unsigned N, unsigned nsuper) {
    unsigned wib = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x, nthr = WN * 32u;
    unsigned tm = blockIdx.x, nbase = blockIdx.y * (WN * 16u);     // this block's M-tile + N-base
    __shared__ signed char na[16][256];                           // weights: staged once, shared by all warps
    __shared__ signed char xs[WN * 16][256];                      // activations: WN token tiles
    __shared__ unsigned char scsh[16][8], msh[16][8];
    __shared__ float drow[16], dmrow[16], dxsh[WN * 16];
    __shared__ int bsh[WN * 16][8];
    __shared__ int probe[WN][16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    // derive element->(row,col) for this warp's accumulator lanes (store-based, portable)
    int rowmap[8], colmap[8];
    for (int e = 0; e < fc.num_elements; e++) fc.x[e] = (int)(lane * 8u + (unsigned)e);
    wmma::store_matrix_sync(&probe[wib][0][0], fc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = 0; e < 8; e++) { rowmap[e] = 0; colmap[e] = 0; }
    for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) { int id = probe[wib][r][c]; if (((unsigned)id >> 3) == lane) { int e = id & 7; rowmap[e] = r; colmap[e] = c; } }
    __syncthreads();

    float outacc[8]; for (int e = 0; e < 8; e++) outacc[e] = 0.0f;
    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = tid; i < 16u * 256u; i += nthr) {       // weights: all threads, once per block
            unsigned r = i >> 8, e = i & 255u, wr = tm * 16u + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            na[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < WN * 16u * 256u; i += nthr) {   // activations: WN token tiles
            unsigned c = i >> 8, e = i & 255u, tc = nbase + c; xs[c][e] = xq[((size_t)tc * nsuper + s) * 256u + e];
        }
        if (tid < 16u) {
            unsigned wr = tm * 16u + tid; const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u;
            for (unsigned sub = 0u; sub < 8u; sub++) { unsigned sv, mv; gsm(sub, scb, &sv, &mv); scsh[tid][sub] = (unsigned char)sv; msh[tid][sub] = (unsigned char)mv; }
            const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u; drow[tid] = f16f(blk[0] | (blk[1] << 8)); dmrow[tid] = f16f(blk[2] | (blk[3] << 8));
        }
        for (unsigned i = tid; i < WN * 16u; i += nthr) { unsigned tc = nbase + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[i][g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        __syncthreads();
        int acc[8], accm[8]; for (int e = 0; e < 8; e++) { acc[e] = 0; accm[e] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned base = (sub >> 1) * 64u + (sub & 1u) * 32u;
            wmma::fill_fragment(fc, 0);
            wmma::load_matrix_sync(fa, &na[0][base], 256);        wmma::load_matrix_sync(fb, &xs[wib * 16u][base], 256);        wmma::mma_sync(fc, fa, fb, fc);
            wmma::load_matrix_sync(fa, &na[0][base + 16u], 256);  wmma::load_matrix_sync(fb, &xs[wib * 16u][base + 16u], 256);  wmma::mma_sync(fc, fa, fb, fc);
            for (int e = 0; e < 8; e++) { int r = rowmap[e], cc = colmap[e]; acc[e] += (int)scsh[r][sub] * fc.x[e]; accm[e] += (int)msh[r][sub] * bsh[wib * 16u + cc][sub]; }
        }
        for (int e = 0; e < 8; e++) { int r = rowmap[e], cc = colmap[e]; float a1 = drow[r] * dxsh[wib * 16u + cc], a2 = dmrow[r] * dxsh[wib * 16u + cc]; outacc[e] += a1 * (float)acc[e] - a2 * (float)accm[e]; }
        __syncthreads();
    }
    for (int e = 0; e < 8; e++) { unsigned wr = tm * 16u + rowmap[e], tc = nbase + wib * 16u + colmap[e]; if (wr < M && tc < N) out[(size_t)wr * N + tc] = outacc[e]; }
}

// ---- v8: wide BM×BN tiled block. The whole block stages BM weight rows AND BN activation cols
// ONCE per super-block and reuses them across all (BM/16)·(BN/16) tiles — amortizing BOTH the
// weight unpack AND the activation staging (v7 re-staged activations once per M-tile; at M=4096
// that was 256× redundant). Each warp owns TPW tiles; in-register per-sub-block scaling as in v6/v7.
// Bit-identical to ref_dot_rs. ----
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BW
#define BW 8
#endif
__global__ void __launch_bounds__(BW * 32) q4k_imma_v8(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                                                       float* out, unsigned M, unsigned N, unsigned nsuper) {
    const unsigned MT = BM / 16u, NT = BN / 16u, NTILES = MT * NT, TPW = NTILES / BW;
    unsigned wib = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x, nthr = BW * 32u;
    unsigned mbase = blockIdx.x * BM, nbase = blockIdx.y * BN;
    __shared__ signed char wsh[BM][256], xsh[BN][256];
    __shared__ unsigned char scsh[BM][8], msh[BM][8];
    __shared__ float drow[BM], dmrow[BM], dxsh[BN];
    __shared__ int bsh[BN][8];
    __shared__ int probe[BW][16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    int rowmap[8], colmap[8];
    for (int e = 0; e < fc.num_elements; e++) fc.x[e] = (int)(lane * 8u + (unsigned)e);
    wmma::store_matrix_sync(&probe[wib][0][0], fc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = 0; e < 8; e++) { rowmap[e] = 0; colmap[e] = 0; }
    for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) { int id = probe[wib][r][c]; if (((unsigned)id >> 3) == lane) { int e = id & 7; rowmap[e] = r; colmap[e] = c; } }
    __syncthreads();

    float outacc[TPW][8]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 8; e++) outacc[i][e] = 0.0f;
    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = tid; i < BM * 256u; i += nthr) {
            unsigned r = i >> 8, e = i & 255u, wr = mbase + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            wsh[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < BN * 256u; i += nthr) { unsigned c = i >> 8, e = i & 255u, tc = nbase + c; xsh[c][e] = xq[((size_t)tc * nsuper + s) * 256u + e]; }
        for (unsigned i = tid; i < BM * 8u; i += nthr) { unsigned r = i >> 3, sub = i & 7u, wr = mbase + r; const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u; unsigned sv, mv; gsm(sub, scb, &sv, &mv); scsh[r][sub] = (unsigned char)sv; msh[r][sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < BM; i += nthr) { unsigned wr = mbase + i; const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u; drow[i] = f16f(blk[0] | (blk[1] << 8)); dmrow[i] = f16f(blk[2] | (blk[3] << 8)); }
        for (unsigned i = tid; i < BN; i += nthr) { unsigned tc = nbase + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[i][g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        __syncthreads();
        int acc[TPW][8], accm[TPW][8]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 8; e++) { acc[i][e] = 0; accm[i][e] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned base = (sub >> 1) * 64u + (sub & 1u) * 32u;
            for (unsigned i = 0; i < TPW; i++) {
                unsigned t = wib * TPW + i, mt = t / NT, nt = t % NT;
                wmma::fill_fragment(fc, 0);
                wmma::load_matrix_sync(fa, &wsh[mt * 16u][base], 256);       wmma::load_matrix_sync(fb, &xsh[nt * 16u][base], 256);       wmma::mma_sync(fc, fa, fb, fc);
                wmma::load_matrix_sync(fa, &wsh[mt * 16u][base + 16u], 256); wmma::load_matrix_sync(fb, &xsh[nt * 16u][base + 16u], 256); wmma::mma_sync(fc, fa, fb, fc);
                for (int e = 0; e < 8; e++) { int r = rowmap[e], cc = colmap[e]; acc[i][e] += (int)scsh[mt * 16u + r][sub] * fc.x[e]; accm[i][e] += (int)msh[mt * 16u + r][sub] * bsh[nt * 16u + cc][sub]; }
            }
        }
        for (unsigned i = 0; i < TPW; i++) {
            unsigned t = wib * TPW + i, mt = t / NT, nt = t % NT;
            for (int e = 0; e < 8; e++) { int r = rowmap[e], cc = colmap[e]; float a1 = drow[mt * 16u + r] * dxsh[nt * 16u + cc], a2 = dmrow[mt * 16u + r] * dxsh[nt * 16u + cc]; outacc[i][e] += a1 * (float)acc[i][e] - a2 * (float)accm[i][e]; }
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < TPW; i++) {
        unsigned t = wib * TPW + i, mt = t / NT, nt = t % NT;
        for (int e = 0; e < 8; e++) { unsigned wr = mbase + mt * 16u + rowmap[e], tc = nbase + nt * 16u + colmap[e]; if (wr < M && tc < N) out[(size_t)wr * N + tc] = outacc[i][e]; }
    }
}

// ---- v9: v8's wide tile, but in DYNAMIC shared memory so BM×BN can exceed the 48 KB static cap
// (opt in via cudaFuncAttributeMaxDynamicSharedMemorySize; sm_80+ allows up to ~100 KB). Wider tiles
// amortize the staging further (weight reuse = BN2/16, activation reuse = BM2/16). Same in-register
// scaling; bit-identical. ----
#ifndef BM2
#define BM2 128
#endif
#ifndef BN2
#define BN2 64
#endif
#ifndef BW2
#define BW2 8
#endif
__global__ void __launch_bounds__(BW2 * 32) q4k_imma_v9(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                                                        float* out, unsigned M, unsigned N, unsigned nsuper) {
    const unsigned MT = BM2 / 16u, NT = BN2 / 16u, NTILES = MT * NT, TPW = NTILES / BW2;
    unsigned wib = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x, nthr = BW2 * 32u;
    unsigned mbase = blockIdx.x * BM2, nbase = blockIdx.y * BN2;
    extern __shared__ unsigned char smem[];
    signed char* wsh = (signed char*)smem;                 // [BM2][256]
    signed char* xsh = wsh + (size_t)BM2 * 256;            // [BN2][256]
    unsigned char* scsh = (unsigned char*)(xsh + (size_t)BN2 * 256); // [BM2][8]
    unsigned char* msh = scsh + (size_t)BM2 * 8;           // [BM2][8]
    float* drow = (float*)(msh + (size_t)BM2 * 8);         // [BM2]
    float* dmrow = drow + BM2;                             // [BM2]
    float* dxsh = dmrow + BM2;                             // [BN2]
    int* bsh = (int*)(dxsh + BN2);                         // [BN2][8]
    int* probe = bsh + (size_t)BN2 * 8;                    // [BW2][256]

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    int rowmap[8], colmap[8];
    for (int e = 0; e < fc.num_elements; e++) fc.x[e] = (int)(lane * 8u + (unsigned)e);
    wmma::store_matrix_sync(probe + (size_t)wib * 256, fc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = 0; e < 8; e++) { rowmap[e] = 0; colmap[e] = 0; }
    for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) { int id = probe[(size_t)wib * 256 + r * 16 + c]; if (((unsigned)id >> 3) == lane) { int e = id & 7; rowmap[e] = r; colmap[e] = c; } }
    __syncthreads();

    float outacc[TPW][8]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 8; e++) outacc[i][e] = 0.0f;
    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = tid; i < BM2 * 256u; i += nthr) {
            unsigned r = i >> 8, e = i & 255u, wr = mbase + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            wsh[(size_t)r * 256 + e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < BN2 * 256u; i += nthr) { unsigned c = i >> 8, e = i & 255u, tc = nbase + c; xsh[(size_t)c * 256 + e] = xq[((size_t)tc * nsuper + s) * 256u + e]; }
        for (unsigned i = tid; i < BM2 * 8u; i += nthr) { unsigned r = i >> 3, sub = i & 7u, wr = mbase + r; const unsigned char* scb = W + (size_t)wr * nsuper * 144u + s * 144u + 4u; unsigned sv, mv; gsm(sub, scb, &sv, &mv); scsh[(size_t)r * 8 + sub] = (unsigned char)sv; msh[(size_t)r * 8 + sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < BM2; i += nthr) { unsigned wr = mbase + i; const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u; drow[i] = f16f(blk[0] | (blk[1] << 8)); dmrow[i] = f16f(blk[2] | (blk[3] << 8)); }
        for (unsigned i = tid; i < BN2; i += nthr) { unsigned tc = nbase + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[(size_t)i * 8 + g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        __syncthreads();
        int acc[TPW][8], accm[TPW][8]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 8; e++) { acc[i][e] = 0; accm[i][e] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned base = (sub >> 1) * 64u + (sub & 1u) * 32u;
            for (unsigned i = 0; i < TPW; i++) {
                unsigned t = wib * TPW + i, mt = t / NT, nt = t % NT;
                wmma::fill_fragment(fc, 0);
                wmma::load_matrix_sync(fa, wsh + (size_t)mt * 16 * 256 + base, 256);       wmma::load_matrix_sync(fb, xsh + (size_t)nt * 16 * 256 + base, 256);       wmma::mma_sync(fc, fa, fb, fc);
                wmma::load_matrix_sync(fa, wsh + (size_t)mt * 16 * 256 + base + 16, 256);   wmma::load_matrix_sync(fb, xsh + (size_t)nt * 16 * 256 + base + 16, 256);   wmma::mma_sync(fc, fa, fb, fc);
                for (int e = 0; e < 8; e++) { int r = rowmap[e], cc = colmap[e]; acc[i][e] += (int)scsh[(size_t)(mt * 16u + r) * 8 + sub] * fc.x[e]; accm[i][e] += (int)msh[(size_t)(mt * 16u + r) * 8 + sub] * bsh[(size_t)(nt * 16u + cc) * 8 + sub]; }
            }
        }
        for (unsigned i = 0; i < TPW; i++) {
            unsigned t = wib * TPW + i, mt = t / NT, nt = t % NT;
            for (int e = 0; e < 8; e++) { int r = rowmap[e], cc = colmap[e]; float a1 = drow[mt * 16u + r] * dxsh[nt * 16u + cc], a2 = dmrow[mt * 16u + r] * dxsh[nt * 16u + cc]; outacc[i][e] += a1 * (float)acc[i][e] - a2 * (float)accm[i][e]; }
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < TPW; i++) {
        unsigned t = wib * TPW + i, mt = t / NT, nt = t % NT;
        for (int e = 0; e < 8; e++) { unsigned wr = mbase + mt * 16u + rowmap[e], tc = nbase + nt * 16u + colmap[e]; if (wr < M && tc < N) out[(size_t)wr * N + tc] = outacc[i][e]; }
    }
}

// ---- v11: mma.sync m16n8k32 (one instruction per Q4_K sub-block, K=32) with in-register scaling.
// WMMA int8 is m16n16k16 (K=16), so a 32-element Q4_K sub-block costs TWO wmma mma + bookkeeping.
// The PTX mma.sync m16n8k32.s8 does the whole sub-block in ONE instruction with half the issue
// count. Wide BMxBN block (v8-style staging). The Ampere s8 fragment layout is fixed/documented:
//   lane -> groupID=lane/4, tid=lane%4
//   A[16][32] rowmaj: a0=rows[gid][tid*4..], a1=rows[gid+8][tid*4..], a2=rows[gid][16+tid*4..], a3=rows[gid+8][16+tid*4..]
//   B[32][8]  colmaj: b0=col[gid] k=tid*4.., b1=col[gid] k=16+tid*4..
//   D[16][8]: d0=(gid,tid*2) d1=(gid,tid*2+1) d2=(gid+8,tid*2) d3=(gid+8,tid*2+1)
// Scales applied to D in registers; running-sum combine; bit-identical to ref_dot_rs. sm_80+. ----
#ifndef MM
#define MM 64
#endif
#ifndef MN
#define MN 64
#endif
#ifndef MW
#define MW 8
#endif
__global__ void __launch_bounds__(MW * 32) q4k_mma_v11(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                                                       float* out, unsigned M, unsigned N, unsigned nsuper) {
    const unsigned VM = MM / 16u, VN = MN / 8u, NTILES = VM * VN, TPW = NTILES / MW;
    unsigned wib = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x, nthr = MW * 32u;
    unsigned gid = lane >> 2, tig = lane & 3u;                 // groupID 0..7, thread-in-group 0..3
    unsigned mbase = blockIdx.x * MM, nbase = blockIdx.y * MN;
    __shared__ signed char wsh[MM][256], xsh[MN][256];
    __shared__ unsigned char scsh[MM][8], msh[MM][8];
    __shared__ float drow[MM], dmrow[MM], dxsh[MN];
    __shared__ int bsh[MN][8];

    float oa[8][4]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 4; e++) oa[i][e] = 0.0f;  // per tile: 4 D regs
    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = tid; i < MM * 256u; i += nthr) {
            unsigned r = i >> 8, e = i & 255u, wr = mbase + r;
            const unsigned char* qs = W + (size_t)wr * nsuper * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            wsh[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < MN * 256u; i += nthr) { unsigned c = i >> 8, e = i & 255u, tc = nbase + c; xsh[c][e] = xq[((size_t)tc * nsuper + s) * 256u + e]; }
        for (unsigned i = tid; i < MM * 8u; i += nthr) { unsigned r = i >> 3, sub = i & 7u, wr = mbase + r; unsigned sv, mv; gsm(sub, W + (size_t)wr * nsuper * 144u + s * 144u + 4u, &sv, &mv); scsh[r][sub] = (unsigned char)sv; msh[r][sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < MM; i += nthr) { const unsigned char* blk = W + (size_t)(mbase + i) * nsuper * 144u + s * 144u; drow[i] = f16f(blk[0] | (blk[1] << 8)); dmrow[i] = f16f(blk[2] | (blk[3] << 8)); }
        for (unsigned i = tid; i < MN; i += nthr) { unsigned tc = nbase + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[i][g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        __syncthreads();
        int acc[8][4], accm[8][4]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 4; e++) { acc[i][e] = 0; accm[i][e] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned base = (sub >> 1) * 64u + (sub & 1u) * 32u;   // this sub-block's 32-elem base
            for (unsigned i = 0; i < TPW; i++) {
                unsigned t = wib * TPW + i, vm = t / VN, vn = t % VN, rbase = vm * 16u, cbase = vn * 8u;
                int a0 = *(const int*)&wsh[rbase + gid][base + tig * 4u];
                int a1 = *(const int*)&wsh[rbase + gid + 8u][base + tig * 4u];
                int a2 = *(const int*)&wsh[rbase + gid][base + 16u + tig * 4u];
                int a3 = *(const int*)&wsh[rbase + gid + 8u][base + 16u + tig * 4u];
                int b0 = *(const int*)&xsh[cbase + gid][base + tig * 4u];
                int b1 = *(const int*)&xsh[cbase + gid][base + 16u + tig * 4u];
                int d0, d1, d2, d3;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
                int scr0 = scsh[rbase + gid][sub], scr2 = scsh[rbase + gid + 8u][sub];
                int mr0 = msh[rbase + gid][sub], mr2 = msh[rbase + gid + 8u][sub];
                int bc0 = bsh[cbase + tig * 2u][sub], bc1 = bsh[cbase + tig * 2u + 1u][sub];
                acc[i][0] += scr0 * d0; acc[i][1] += scr0 * d1; acc[i][2] += scr2 * d2; acc[i][3] += scr2 * d3;
                accm[i][0] += mr0 * bc0; accm[i][1] += mr0 * bc1; accm[i][2] += mr2 * bc0; accm[i][3] += mr2 * bc1;
            }
        }
        for (unsigned i = 0; i < TPW; i++) {
            unsigned t = wib * TPW + i, vm = t / VN, vn = t % VN, rbase = vm * 16u, cbase = vn * 8u;
            float dr0 = drow[rbase + gid], dm0 = dmrow[rbase + gid], dr2 = drow[rbase + gid + 8u], dm2 = dmrow[rbase + gid + 8u];
            float dxc0 = dxsh[cbase + tig * 2u], dxc1 = dxsh[cbase + tig * 2u + 1u];
            oa[i][0] += dr0 * dxc0 * (float)acc[i][0] - dm0 * dxc0 * (float)accm[i][0];
            oa[i][1] += dr0 * dxc1 * (float)acc[i][1] - dm0 * dxc1 * (float)accm[i][1];
            oa[i][2] += dr2 * dxc0 * (float)acc[i][2] - dm2 * dxc0 * (float)accm[i][2];
            oa[i][3] += dr2 * dxc1 * (float)acc[i][3] - dm2 * dxc1 * (float)accm[i][3];
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < TPW; i++) {
        unsigned t = wib * TPW + i, vm = t / VN, vn = t % VN, rbase = mbase + vm * 16u, cbase = nbase + vn * 8u;
        unsigned r0 = rbase + gid, r2 = rbase + gid + 8u, c0 = cbase + tig * 2u, c1 = cbase + tig * 2u + 1u;
        if (r0 < M && c0 < N) out[(size_t)r0 * N + c0] = oa[i][0];
        if (r0 < M && c1 < N) out[(size_t)r0 * N + c1] = oa[i][1];
        if (r2 < M && c0 < N) out[(size_t)r2 * N + c0] = oa[i][2];
        if (r2 < M && c1 < N) out[(size_t)r2 * N + c1] = oa[i][3];
    }
}

// ---- v12: v11's mma.sync, now with cp.async DOUBLE-BUFFERED staging to hide the global-read
// latency (the staging-bound ceiling). Raw packed weights + int8 activations for super-block s+1 are
// async-copied into a second shared buffer while super-block s computes; the cheap nibble unpack then
// runs shared->shared. Dynamic shared (sm_80+). Bit-identical to ref_dot_rs. ----
#ifndef DM
#define DM 64
#endif
#ifndef DN
#define DN 64
#endif
#ifndef DW
#define DW 8
#endif
__global__ void __launch_bounds__(DW * 32) q4k_mma_v12(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                                                       float* out, unsigned M, unsigned N, unsigned nsuper) {
    const unsigned VM = DM / 16u, VN = DN / 8u, NTILES = VM * VN, TPW = NTILES / DW;
    unsigned wib = threadIdx.x >> 5, lane = threadIdx.x & 31u, tid = threadIdx.x, nthr = DW * 32u;
    unsigned gid = lane >> 2, tig = lane & 3u;
    unsigned mbase = blockIdx.x * DM, nbase = blockIdx.y * DN;
    extern __shared__ unsigned char smem[];
    unsigned char* rw = smem;                                   // [2][DM][144] raw packed weights
    signed char* xs = (signed char*)(rw + (size_t)2 * DM * 144);// [2][DN][256] int8 activations
    signed char* wsh = xs + (size_t)2 * DN * 256;               // [DM][256] unpacked nibbles
    unsigned char* scsh = (unsigned char*)(wsh + (size_t)DM * 256); // [DM][8]
    unsigned char* msh = scsh + (size_t)DM * 8;                 // [DM][8]
    float* drow = (float*)(msh + (size_t)DM * 8);               // [DM]
    float* dmrow = drow + DM;                                   // [DM]
    float* dxsh = dmrow + DM;                                   // [DN]
    int* bsh = (int*)(dxsh + DN);                               // [DN][8]

    // async-load super-block s's raw weights + activations into buffer `buf`
    auto load = [&](unsigned buf, unsigned s) {
        unsigned char* rwb = rw + (size_t)buf * DM * 144;
        for (unsigned i = tid; i < DM * 9u; i += nthr) { unsigned r = i / 9u, w = (i % 9u) * 16u; __pipeline_memcpy_async(rwb + (size_t)r * 144 + w, W + (size_t)(mbase + r) * nsuper * 144u + s * 144u + w, 16); }
        signed char* xsb = xs + (size_t)buf * DN * 256;
        for (unsigned i = tid; i < DN * 16u; i += nthr) { unsigned c = i >> 4, w = (i & 15u) * 16u; __pipeline_memcpy_async(xsb + (size_t)c * 256 + w, xq + ((size_t)(nbase + c) * nsuper + s) * 256u + w, 16); }
        __pipeline_commit();
    };

    float oa[8][4]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 4; e++) oa[i][e] = 0.0f;
    load(0, 0);
    for (unsigned s = 0u; s < nsuper; s++) {
        unsigned cur = s & 1u;
        if (s + 1u < nsuper) load((s + 1u) & 1u, s + 1u);
        __pipeline_wait_prior(s + 1u < nsuper ? 1 : 0);
        __syncthreads();
        // unpack raw -> wsh + scales/d (shared -> shared), stage dx/bsums from global
        unsigned char* rwb = rw + (size_t)cur * DM * 144;
        for (unsigned i = tid; i < DM * 256u; i += nthr) {
            unsigned r = i >> 8, e = i & 255u; const unsigned char* qs = rwb + (size_t)r * 144 + 16;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            wsh[(size_t)r * 256 + e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < DM * 8u; i += nthr) { unsigned r = i >> 3, sub = i & 7u; unsigned sv, mv; gsm(sub, rwb + (size_t)r * 144 + 4, &sv, &mv); scsh[(size_t)r * 8 + sub] = (unsigned char)sv; msh[(size_t)r * 8 + sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < DM; i += nthr) { const unsigned char* blk = rwb + (size_t)i * 144; drow[i] = f16f(blk[0] | (blk[1] << 8)); dmrow[i] = f16f(blk[2] | (blk[3] << 8)); }
        for (unsigned i = tid; i < DN; i += nthr) { unsigned tc = nbase + i; dxsh[i] = dx[(size_t)tc * nsuper + s]; for (int g = 0; g < 8; g++) bsh[(size_t)i * 8 + g] = bsums[((size_t)tc * nsuper + s) * 8u + g]; }
        __syncthreads();
        int acc[8][4], accm[8][4]; for (unsigned i = 0; i < TPW; i++) for (int e = 0; e < 4; e++) { acc[i][e] = 0; accm[i][e] = 0; }
        signed char* xsb = xs + (size_t)cur * DN * 256;
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned base = (sub >> 1) * 64u + (sub & 1u) * 32u;
            for (unsigned i = 0; i < TPW; i++) {
                unsigned t = wib * TPW + i, vm = t / VN, vn = t % VN, rb = vm * 16u, cb = vn * 8u;
                int a0 = *(const int*)&wsh[(size_t)(rb + gid) * 256 + base + tig * 4u];
                int a1 = *(const int*)&wsh[(size_t)(rb + gid + 8u) * 256 + base + tig * 4u];
                int a2 = *(const int*)&wsh[(size_t)(rb + gid) * 256 + base + 16u + tig * 4u];
                int a3 = *(const int*)&wsh[(size_t)(rb + gid + 8u) * 256 + base + 16u + tig * 4u];
                int b0 = *(const int*)&xsb[(size_t)(cb + gid) * 256 + base + tig * 4u];
                int b1 = *(const int*)&xsb[(size_t)(cb + gid) * 256 + base + 16u + tig * 4u];
                int d0, d1, d2, d3;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
                int scr0 = scsh[(size_t)(rb + gid) * 8 + sub], scr2 = scsh[(size_t)(rb + gid + 8u) * 8 + sub];
                int mr0 = msh[(size_t)(rb + gid) * 8 + sub], mr2 = msh[(size_t)(rb + gid + 8u) * 8 + sub];
                int bc0 = bsh[(size_t)(cb + tig * 2u) * 8 + sub], bc1 = bsh[(size_t)(cb + tig * 2u + 1u) * 8 + sub];
                acc[i][0] += scr0 * d0; acc[i][1] += scr0 * d1; acc[i][2] += scr2 * d2; acc[i][3] += scr2 * d3;
                accm[i][0] += mr0 * bc0; accm[i][1] += mr0 * bc1; accm[i][2] += mr2 * bc0; accm[i][3] += mr2 * bc1;
            }
        }
        for (unsigned i = 0; i < TPW; i++) {
            unsigned t = wib * TPW + i, vm = t / VN, vn = t % VN, rb = vm * 16u, cb = vn * 8u;
            float dr0 = drow[rb + gid], dm0 = dmrow[rb + gid], dr2 = drow[rb + gid + 8u], dm2 = dmrow[rb + gid + 8u];
            float dxc0 = dxsh[cb + tig * 2u], dxc1 = dxsh[cb + tig * 2u + 1u];
            oa[i][0] += dr0 * dxc0 * (float)acc[i][0] - dm0 * dxc0 * (float)accm[i][0];
            oa[i][1] += dr0 * dxc1 * (float)acc[i][1] - dm0 * dxc1 * (float)accm[i][1];
            oa[i][2] += dr2 * dxc0 * (float)acc[i][2] - dm2 * dxc0 * (float)accm[i][2];
            oa[i][3] += dr2 * dxc1 * (float)acc[i][3] - dm2 * dxc1 * (float)accm[i][3];
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < TPW; i++) {
        unsigned t = wib * TPW + i, vm = t / VN, vn = t % VN, rb = mbase + vm * 16u, cb = nbase + vn * 8u;
        unsigned r0 = rb + gid, r2 = rb + gid + 8u, c0 = cb + tig * 2u, c1 = cb + tig * 2u + 1u;
        if (r0 < M && c0 < N) out[(size_t)r0 * N + c0] = oa[i][0];
        if (r0 < M && c1 < N) out[(size_t)r0 * N + c1] = oa[i][1];
        if (r2 < M && c0 < N) out[(size_t)r2 * N + c0] = oa[i][2];
        if (r2 < M && c1 < N) out[(size_t)r2 * N + c1] = oa[i][3];
    }
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
// ---- v3: co-optimized multi-warp block. One block = VWARPS warps owns 16 weight-rows ×
// (VWARPS*16) token-columns; all warps SHARE one shared-memory weight staging (weights read
// once per col-group, not per tile), each warp drives its own N-tile's WMMA + epilogue, and
// the grid keeps thousands of warps live (occupancy). Writes per-super-block parts to global;
// q4k_combine does the fixed tree. Bit-identical to v1/v2. ----
#ifndef VWARPS
#define VWARPS 8
#endif
__global__ void __launch_bounds__(VWARPS * 32) q4k_imma_v3(
        const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
        float* parts, unsigned M, unsigned N, unsigned nsuper) {
    unsigned tm = blockIdx.x; if (tm * 16u >= M) return;
    unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    unsigned nt = blockIdx.y * VWARPS + warp;            // this warp's token-tile
    bool active = (nt * 16u < N);
    extern __shared__ unsigned char smem[];
    signed char* na = (signed char*)smem;                        // [16][32] shared weights
    unsigned* scrow = (unsigned*)(na + 16 * 32);                 // [16]
    unsigned* mrow = scrow + 16;                                 // [16]
    signed char* nbuf = (signed char*)(mrow + 16);               // [VWARPS][16][32]
    int* sjt = (int*)(nbuf + VWARPS * 16 * 32);                  // [VWARPS][256]
    int* acc = sjt + VWARPS * 256;                               // [VWARPS][256]
    int* accm = acc + VWARPS * 256;                              // [VWARPS][256]
    signed char* mynb = nbuf + warp * 512; int* mysj = sjt + warp * 256; int* myacc = acc + warp * 256; int* myam = accm + warp * 256;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;

    for (unsigned s = 0u; s < nsuper; s++) {
        for (unsigned i = lane; i < 256u; i += 32u) { myacc[i] = 0; myam[i] = 0; }
        for (unsigned sub = 0u; sub < 8u; sub++) {
            unsigned half = sub >> 1, hi = sub & 1u, qoff = half * 32u, base = half * 64u + hi * 32u;
            for (unsigned i = threadIdx.x; i < 16u * 32u; i += blockDim.x) {   // block-cooperative weight stage (ONCE/colgroup)
                unsigned r = i >> 5, l = i & 31u; const unsigned char* qs = W + (size_t)(tm * 16u + r) * nsuper * 144u + s * 144u + 16u;
                unsigned q = qs[qoff + l]; na[i] = (signed char)(hi ? (q >> 4) : (q & 0x0Fu));
            }
            if (threadIdx.x < 16u) { const unsigned char* scb = W + (size_t)(tm * 16u + threadIdx.x) * nsuper * 144u + s * 144u + 4u; gsm(sub, scb, &scrow[threadIdx.x], &mrow[threadIdx.x]); }
            __syncthreads();
            if (active) {
                for (unsigned i = lane; i < 16u * 32u; i += 32u) { unsigned c = i >> 5, l = i & 31u, tc = nt * 16u + c; mynb[c * 32u + l] = xq[((size_t)tc * nsuper + s) * 256u + base + l]; }
                __syncwarp();
                wmma::fill_fragment(fc, 0);
                wmma::load_matrix_sync(fa, na, 32); wmma::load_matrix_sync(fb, mynb, 32); wmma::mma_sync(fc, fa, fb, fc);
                wmma::load_matrix_sync(fa, na + 16, 32); wmma::load_matrix_sync(fb, mynb + 16, 32); wmma::mma_sync(fc, fa, fb, fc);
                wmma::store_matrix_sync(mysj, fc, 16, wmma::mem_row_major);
                __syncwarp();
                for (unsigned i = lane; i < 256u; i += 32u) { unsigned r = i >> 4, c = i & 15u, tc = nt * 16u + c; myacc[i] += (int)scrow[r] * mysj[i]; myam[i] += (int)mrow[r] * bsums[((size_t)tc * nsuper + s) * 8u + sub]; }
            }
            __syncthreads();                                   // all warps done with na before next sub restages it
        }
        if (active) {
            for (unsigned i = lane; i < 256u; i += 32u) {
                unsigned r = i >> 4, c = i & 15u, wr = tm * 16u + r, tc = nt * 16u + c;
                const unsigned char* blk = W + (size_t)wr * nsuper * 144u + s * 144u;
                float d = f16f(blk[0] | (blk[1] << 8)), dmin = f16f(blk[2] | (blk[3] << 8)), dxv = dx[(size_t)tc * nsuper + s];
                float a1 = d * dxv, a2 = dmin * dxv;
                parts[((size_t)wr * N + tc) * nsuper + s] = a1 * (float)myacc[i] - a2 * (float)myam[i];
            }
        }
        __syncthreads();
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

    // ---- v3: multi-warp block (shared weight staging + occupancy) ----
    if (N % (VWARPS * 16) == 0) {
        size_t sh3 = 16 * 32 + 16 * 4 + 16 * 4 + VWARPS * (16 * 32 + 256 * 4 * 3);
        cudaFuncSetAttribute(q4k_imma_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh3);
        dim3 grid3(M / 16, N / (VWARPS * 16)); dim3 blk3(VWARPS * 32);
        q4k_imma_v3<<<grid3, blk3, sh3>>>(dW, ddx, dqs, dbs, dparts, M, N, nsuper);
        q4k_combine<<<((size_t)M * N + 127) / 128, 128>>>(dparts, dout, M, N, nsuper);
        cudaError_t e3 = cudaDeviceSynchronize();
        if (e3 != cudaSuccess) { printf("v3 launch error: %s (sh3=%zu)\n", cudaGetErrorString(e3), sh3); return 1; }
        float* gpu3 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu3, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        size_t d3 = 0, f3 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &ref[i], 4); memcpy(&b, &gpu3[i], 4); if (a != b) { if (f3 == (size_t)-1) f3 = i; d3++; } }
        printf("v3 multiwarp bit-identity: %zu/%d diffs%s", d3, M * N, d3 ? "" : "  -> BIT-IDENTICAL\n");
        if (d3) printf("   first @ %zu ref=%g gpu=%g\n", f3, ref[f3], gpu3[f3]);
        cudaEventRecord(s0); for (int it = 0; it < iters; it++) { q4k_imma_v3<<<grid3, blk3, sh3>>>(dW, ddx, dqs, dbs, dparts, M, N, nsuper); q4k_combine<<<((size_t)M * N + 127) / 128, 128>>>(dparts, dout, M, N, nsuper); }
        cudaEventRecord(e0); cudaEventSynchronize(e0); float ms3 = 0; cudaEventElapsedTime(&ms3, s0, e0); ms3 /= iters;
        printf("v3 multiwarp(VWARPS=%d) : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1)\n", VWARPS, ms3, 2.0 * M * N * K / (ms3 * 1e-3) / 1e12, ms / ms3);
    }

    // ---- v4: register-tiled dp4a (running-sum regime) ----
    if (M % TBM == 0 && N % TBN == 0) {
        float* refrs = (float*)malloc((size_t)M * N * 4);
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++)
            refrs[(size_t)m * N + n] = ref_dot_rs(hW + (size_t)m * nsuper * 144, dxh + (size_t)n * nsuper, qsh + (size_t)n * nsuper * 256, bsh + (size_t)n * nsuper * 8, nsuper);
        dim3 grid4(M / TBM, N / TBN); dim3 blk4(256);
        q4k_dp4a_v4<<<grid4, blk4>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
        cudaError_t e4 = cudaDeviceSynchronize();
        if (e4 != cudaSuccess) { printf("v4 launch error: %s\n", cudaGetErrorString(e4)); return 1; }
        float* gpu4 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu4, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        size_t d4 = 0, f4 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu4[i], 4); if (a != b) { if (f4 == (size_t)-1) f4 = i; d4++; } }
        printf("v4 dp4a bit-identity (vs running-sum ref): %zu/%d diffs%s", d4, M * N, d4 ? "" : "  -> BIT-IDENTICAL\n");
        if (d4) printf("   first @ %zu refrs=%g gpu=%g\n", f4, refrs[f4], gpu4[f4]);
        cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_dp4a_v4<<<grid4, blk4>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
        cudaEventRecord(e0); cudaEventSynchronize(e0); float ms4 = 0; cudaEventElapsedTime(&ms4, s0, e0); ms4 /= iters;
        printf("v4 dp4a-tiled  : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1)\n", ms4, 2.0 * M * N * K / (ms4 * 1e-3) / 1e12, ms / ms4);
    }

    // ---- v5: on-spec production kernel (register-tiled dp4a pass-1 + fixed_tree combine) ----
    // Validated against the FIXED_TREE reference `ref` (== engine's linear_q4_k_integer).
    if (M % TBM == 0 && N % TBN == 0) {
        dim3 grid5(M / TBM, N / TBN); dim3 blk5(256);
        q4k_dp4a_v5_p1<<<grid5, blk5>>>(dW, ddx, dqs, dbs, dparts, M, N, nsuper);
        q4k_combine<<<((size_t)M * N + 127) / 128, 128>>>(dparts, dout, M, N, nsuper);
        cudaError_t e5 = cudaDeviceSynchronize();
        if (e5 != cudaSuccess) { printf("v5 launch error: %s\n", cudaGetErrorString(e5)); return 1; }
        float* gpu5 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu5, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        size_t d5 = 0, f5 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &ref[i], 4); memcpy(&b, &gpu5[i], 4); if (a != b) { if (f5 == (size_t)-1) f5 = i; d5++; } }
        printf("v5 dp4a+fixedtree bit-identity (vs FIXED-TREE ref = engine regime-3): %zu/%d diffs%s", d5, M * N, d5 ? "" : "  -> BIT-IDENTICAL\n");
        if (d5) printf("   first @ %zu ref=%g gpu=%g\n", f5, ref[f5], gpu5[f5]);
        cudaEventRecord(s0); for (int it = 0; it < iters; it++) { q4k_dp4a_v5_p1<<<grid5, blk5>>>(dW, ddx, dqs, dbs, dparts, M, N, nsuper); q4k_combine<<<((size_t)M * N + 127) / 128, 128>>>(dparts, dout, M, N, nsuper); }
        cudaEventRecord(e0); cudaEventSynchronize(e0); float ms5 = 0; cudaEventElapsedTime(&ms5, s0, e0); ms5 /= iters;
        printf("v5 dp4a+fixedtree : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1)\n", ms5, 2.0 * M * N * K / (ms5 * 1e-3) / 1e12, ms / ms5);
    }

    // ---- v6: WMMA tensor cores, IN-REGISTER per-sub-block scaling (epilogue-bound problem solved) ----
    // Running-sum regime → validate against refrs. One warp per 16x16 tile; static shared.
    if (M % 16 == 0 && N % 16 == 0) {
        float* refrs = (float*)malloc((size_t)M * N * 4);
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++)
            refrs[(size_t)m * N + n] = ref_dot_rs(hW + (size_t)m * nsuper * 144, dxh + (size_t)n * nsuper, qsh + (size_t)n * nsuper * 256, bsh + (size_t)n * nsuper * 8, nsuper);
        unsigned wtot = (M / 16u) * (N / 16u), tpb = 32u, blocks = wtot;  // one warp (tile) per block
        q4k_imma_v6<<<blocks, tpb>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
        cudaError_t e6 = cudaDeviceSynchronize();
        if (e6 != cudaSuccess) { printf("v6 launch error: %s\n", cudaGetErrorString(e6)); return 1; }
        float* gpu6 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu6, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        size_t d6 = 0, f6 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu6[i], 4); if (a != b) { if (f6 == (size_t)-1) f6 = i; d6++; } }
        printf("v6 WMMA-inreg bit-identity (vs running-sum ref): %zu/%d diffs%s", d6, M * N, d6 ? "" : "  -> BIT-IDENTICAL\n");
        if (d6) printf("   first @ %zu refrs=%g gpu=%g\n", f6, refrs[f6], gpu6[f6]);
        cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_imma_v6<<<blocks, tpb>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
        cudaEventRecord(e0); cudaEventSynchronize(e0); float ms6 = 0; cudaEventElapsedTime(&ms6, s0, e0); ms6 /= iters;
        printf("v6 WMMA-inreg  : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1)\n", ms6, 2.0 * M * N * K / (ms6 * 1e-3) / 1e12, ms / ms6);

        // ---- v7: cooperative multi-warp block (weight reuse) ----
        if (N % (WN * 16) == 0) {
            dim3 grid7(M / 16u, N / (WN * 16u)); unsigned tpb7 = WN * 32u;
            q4k_imma_v7<<<grid7, tpb7>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaError_t e7 = cudaDeviceSynchronize();
            if (e7 != cudaSuccess) { printf("v7 launch error: %s\n", cudaGetErrorString(e7)); return 1; }
            float* gpu7 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu7, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
            size_t d7 = 0, f7 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu7[i], 4); if (a != b) { if (f7 == (size_t)-1) f7 = i; d7++; } }
            printf("v7 WMMA-coop bit-identity (vs running-sum ref): %zu/%d diffs%s", d7, M * N, d7 ? "" : "  -> BIT-IDENTICAL\n");
            if (d7) printf("   first @ %zu refrs=%g gpu=%g\n", f7, refrs[f7], gpu7[f7]);
            cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_imma_v7<<<grid7, tpb7>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaEventRecord(e0); cudaEventSynchronize(e0); float ms7 = 0; cudaEventElapsedTime(&ms7, s0, e0); ms7 /= iters;
            printf("v7 WMMA-coop(WN=%d) : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1, %.2fx vs v4-ref)\n", WN, ms7, 2.0 * M * N * K / (ms7 * 1e-3) / 1e12, ms / ms7, ms6 / ms7);
            free(gpu7);
        }

        // ---- v8: wide BM×BN tiled block (amortizes weight AND activation staging) ----
        if (M % BM == 0 && N % BN == 0) {
            dim3 grid8(M / BM, N / BN); unsigned tpb8 = BW * 32u;
            q4k_imma_v8<<<grid8, tpb8>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaError_t e8 = cudaDeviceSynchronize();
            if (e8 != cudaSuccess) { printf("v8 launch error: %s\n", cudaGetErrorString(e8)); return 1; }
            float* gpu8 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu8, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
            size_t d8 = 0, f8 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu8[i], 4); if (a != b) { if (f8 == (size_t)-1) f8 = i; d8++; } }
            printf("v8 WMMA-wide bit-identity (vs running-sum ref): %zu/%d diffs%s", d8, M * N, d8 ? "" : "  -> BIT-IDENTICAL\n");
            if (d8) printf("   first @ %zu refrs=%g gpu=%g\n", f8, refrs[f8], gpu8[f8]);
            cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_imma_v8<<<grid8, tpb8>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaEventRecord(e0); cudaEventSynchronize(e0); float ms8 = 0; cudaEventElapsedTime(&ms8, s0, e0); ms8 /= iters;
            printf("v8 WMMA-wide(%dx%d,BW=%d) : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1, %.2fx vs v4-ref)\n", BM, BN, BW, ms8, 2.0 * M * N * K / (ms8 * 1e-3) / 1e12, ms / ms8, ms6 / ms8);
            free(gpu8);
        }

        // ---- v9: dynamic-shared wide tile (>48KB blocks) ----
        if (M % BM2 == 0 && N % BN2 == 0) {
            size_t shmem9 = (size_t)BM2 * 256 + (size_t)BN2 * 256 + (size_t)BM2 * 16 + (size_t)BM2 * 8 + (size_t)BN2 * 4 + (size_t)BN2 * 32 + (size_t)BW2 * 1024;
            cudaFuncSetAttribute(q4k_imma_v9, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem9);
            dim3 grid9(M / BM2, N / BN2); unsigned tpb9 = BW2 * 32u;
            q4k_imma_v9<<<grid9, tpb9, shmem9>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaError_t e9 = cudaDeviceSynchronize();
            if (e9 != cudaSuccess) { printf("v9 launch error: %s (shmem=%zu)\n", cudaGetErrorString(e9), shmem9); }
            else {
                float* gpu9 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu9, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
                size_t d9 = 0, f9 = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu9[i], 4); if (a != b) { if (f9 == (size_t)-1) f9 = i; d9++; } }
                printf("v9 WMMA-dyn bit-identity (vs running-sum ref): %zu/%d diffs%s", d9, M * N, d9 ? "" : "  -> BIT-IDENTICAL\n");
                if (d9) printf("   first @ %zu refrs=%g gpu=%g\n", f9, refrs[f9], gpu9[f9]);
                cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_imma_v9<<<grid9, tpb9, shmem9>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
                cudaEventRecord(e0); cudaEventSynchronize(e0); float ms9 = 0; cudaEventElapsedTime(&ms9, s0, e0); ms9 /= iters;
                printf("v9 WMMA-dyn(%dx%d,BW=%d,%zuKB) : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1, %.2fx vs v4-ref)\n", BM2, BN2, BW2, shmem9 / 1024, ms9, 2.0 * M * N * K / (ms9 * 1e-3) / 1e12, ms / ms9, ms6 / ms9);
                free(gpu9);
            }
        }

        // ---- v11: mma.sync m16n8k32 (K=32/instruction) ----
        if (p.major >= 8 && M % MM == 0 && N % MN == 0) {
            dim3 grid11(M / MM, N / MN); unsigned tpb11 = MW * 32u;
            q4k_mma_v11<<<grid11, tpb11>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaError_t e11 = cudaDeviceSynchronize();
            if (e11 != cudaSuccess) { printf("v11 launch error: %s\n", cudaGetErrorString(e11)); }
            else {
                float* gpu11 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu11, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
                size_t da = 0, fa = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu11[i], 4); if (a != b) { if (fa == (size_t)-1) fa = i; da++; } }
                printf("v11 mma.sync bit-identity (vs running-sum ref): %zu/%d diffs%s", da, M * N, da ? "" : "  -> BIT-IDENTICAL\n");
                if (da) printf("   first @ %zu refrs=%g gpu=%g\n", fa, refrs[fa], gpu11[fa]);
                cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_mma_v11<<<grid11, tpb11>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
                cudaEventRecord(e0); cudaEventSynchronize(e0); float ms11 = 0; cudaEventElapsedTime(&ms11, s0, e0); ms11 /= iters;
                printf("v11 mma.sync(%dx%d,MW=%d) : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1, %.2fx vs v4-ref)\n", MM, MN, MW, ms11, 2.0 * M * N * K / (ms11 * 1e-3) / 1e12, ms / ms11, ms6 / ms11);
                free(gpu11);
            }
        }

        // ---- v12: mma.sync + cp.async double-buffered staging ----
        if (p.major >= 8 && M % DM == 0 && N % DN == 0) {
            size_t sh12 = (size_t)2 * DM * 144 + (size_t)2 * DN * 256 + (size_t)DM * 256 + (size_t)DM * 16 + (size_t)DM * 8 + (size_t)DN * 4 + (size_t)DN * 32;
            cudaFuncSetAttribute(q4k_mma_v12, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh12);
            dim3 grid12(M / DM, N / DN); unsigned tpb12 = DW * 32u;
            q4k_mma_v12<<<grid12, tpb12, sh12>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
            cudaError_t e12 = cudaDeviceSynchronize();
            if (e12 != cudaSuccess) { printf("v12 launch error: %s (shmem=%zu)\n", cudaGetErrorString(e12), sh12); }
            else {
                float* gpu12 = (float*)malloc((size_t)M * N * 4); cudaMemcpy(gpu12, dout, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
                size_t db = 0, fb = (size_t)-1; for (size_t i = 0; i < (size_t)M * N; i++) { unsigned a, b; memcpy(&a, &refrs[i], 4); memcpy(&b, &gpu12[i], 4); if (a != b) { if (fb == (size_t)-1) fb = i; db++; } }
                printf("v12 mma+cp.async bit-identity (vs running-sum ref): %zu/%d diffs%s", db, M * N, db ? "" : "  -> BIT-IDENTICAL\n");
                if (db) printf("   first @ %zu refrs=%g gpu=%g\n", fb, refrs[fb], gpu12[fb]);
                cudaEventRecord(s0); for (int it = 0; it < iters; it++) q4k_mma_v12<<<grid12, tpb12, sh12>>>(dW, ddx, dqs, dbs, dout, M, N, nsuper);
                cudaEventRecord(e0); cudaEventSynchronize(e0); float ms12 = 0; cudaEventElapsedTime(&ms12, s0, e0); ms12 /= iters;
                printf("v12 mma+cp.async(%dx%d,DW=%d,%zuKB) : %.3f ms  %.1f int8-TOPS   (%.2fx vs v1, %.2fx vs v4-ref)\n", DM, DN, DW, sh12 / 1024, ms12, 2.0 * M * N * K / (ms12 * 1e-3) / 1e12, ms / ms12, ms6 / ms12);
                free(gpu12);
            }
        }
        free(refrs); free(gpu6);
    }
    return 0;
}
