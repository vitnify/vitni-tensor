// vt_cuda.cu — deterministic CUDA inference driver for vitni-tensor.
//
// The full quantized Llama/Mistral/Qwen2 forward on an NVIDIA GPU, bit-for-bit
// identical to the CPU engine. The kernels are the ones proven bit-exact on the
// T4 in kernels/cuda_forward_conformance.cu (div_sw, vt_expf, sqrt, q4k_linear,
// q6k_integer, rms_norm, attention) plus the forward-driver kernels the Metal
// backend uses (q4k_dequant for the embedding, the accumulating linear variants
// that fuse the residual, silu_mul, rope_apply).
//
// Determinism contract: compile with --fmad=false (no mul+add -> fma contraction;
// matches the CPU's separately-rounded mul/add) and -ftz=false (IEEE denormals).
// Because the final SASS is baked by nvcc with these flags, there is no PTX JIT
// step that could re-introduce an FMA. One (default) stream serializes the
// kernels, giving the same data dependency as the Metal backend's barriers.
//
// Build: nvcc --fmad=false -ftz=false -O2 -arch=sm_XX -c vt_cuda.cu
//
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================ device helpers ============================
// (verbatim from kernels/cuda_forward_conformance.cu — proven bit-exact on T4)

__device__ __forceinline__ float div_sw(float a, float b) {
    unsigned sgn = (__float_as_uint(a) ^ __float_as_uint(b)) & 0x80000000u;
    float ba = __uint_as_float(__float_as_uint(b) & 0x7fffffffu);
    float aa = __uint_as_float(__float_as_uint(a) & 0x7fffffffu);
    unsigned j = 0x7EF127EAu - __float_as_uint(ba);
    float y = __uint_as_float(j);
    float nb = -ba;
    float e;
    e = fmaf(nb, y, 1.0f); y = fmaf(y, e, y);
    e = fmaf(nb, y, 1.0f); y = fmaf(y, e, y);
    e = fmaf(nb, y, 1.0f); y = fmaf(y, e, y);
    float q = aa * y;
    float r = fmaf(nb, q, aa);
    q = fmaf(r, y, q);
    return __uint_as_float(__float_as_uint(q) ^ sgn);
}
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    unsigned sign = (unsigned)((h >> 15) & 0x1u);
    unsigned expo = (unsigned)((h >> 10) & 0x1Fu);
    unsigned mant = (unsigned)(h & 0x3FFu);
    unsigned bits;
    if (expo == 0u) {
        if (mant == 0u) bits = sign << 31;
        else { unsigned m = mant; int e = 1; while ((m & 0x400u) == 0u) { m <<= 1; e -= 1; } m &= 0x3FFu; bits = (sign << 31) | ((unsigned)(127 - 15 + e) << 23) | (m << 13); }
    } else if (expo == 0x1Fu) bits = (sign << 31) | 0x7F800000u | (mant << 13);
    else bits = (sign << 31) | ((expo + 127u - 15u) << 23) | (mant << 13);
    return __uint_as_float(bits);
}
__device__ __forceinline__ float fixed_tree(float* part, unsigned len) {
    if (len == 0u) return 0.0f;
    while (len > 1u) { unsigned hl = (len + 1u) / 2u; for (unsigned t = 0u; t < hl; t++) { unsigned u = 2u * t; part[t] = (u + 1u < len) ? (part[u] + part[u + 1u]) : part[u]; } len = hl; }
    return part[0];
}
__device__ __forceinline__ float vt_scalbnf(float x, int n) {
    float f_exp_max = __uint_as_float(254u << 23), f_exp_min = __uint_as_float(1u << 23), f_pow_subnorm = __uint_as_float(151u << 23);
    if (n > 127) { x = x * f_exp_max; n -= 127; if (n > 127) { x = x * f_exp_max; n -= 127; if (n > 127) n = 127; } }
    else if (n < -126) { float mul = f_exp_min * f_pow_subnorm; int add = 126 - 24; x = x * mul; n += add; if (n < -126) { x = x * mul; n += add; if (n < -126) n = -126; } }
    return x * __uint_as_float((unsigned)(127 + n) << 23);
}
__device__ __forceinline__ float vt_expf(float x) {
    const float HALF0 = 0.5f, HALF1 = -0.5f, LN2_HI = 6.9314575195e-01f, LN2_LO = 1.4286067653e-06f, INV_LN2 = 1.4426950216e+00f, P1 = 1.6666625440e-1f, P2 = -2.7667332906e-3f;
    float x1p127 = __uint_as_float(0x7f000000u);
    unsigned hx = __float_as_uint(x);
    int sign = (int)(hx >> 31); bool signb = sign != 0; hx &= 0x7fffffffu;
    if (hx >= 0x42aeac50u) { if (hx > 0x7f800000u) return x; if ((hx >= 0x42b17218u) && !signb) { return x * x1p127; } if (signb) { if (hx >= 0x42cff1b5u) return 0.0f; } }
    int k; float hi, lo;
    if (hx > 0x3eb17218u) {
        if (hx > 0x3f851592u) { float h = signb ? HALF1 : HALF0; k = (int)(INV_LN2 * x + h); } else { k = 1 - sign - sign; }
        float kf = (float)k; hi = x - kf * LN2_HI; lo = kf * LN2_LO; x = hi - lo;
    } else if (hx > 0x39000000u) { k = 0; hi = x; lo = 0.0f; }
    else return 1.0f + x;
    float xx = x * x;
    float c = x - xx * (P1 + xx * P2);   // --fmad=false keeps this un-contracted
    float y = 1.0f + ((div_sw(x * c, 2.0f - c) - lo) + hi);
    return (k == 0) ? y : vt_scalbnf(y, k);
}

// (scale,min) 6-bit unpack — exact port of get_scale_min_k4.
__device__ __forceinline__ void get_scale_min_k4(unsigned j, const unsigned char* sc, unsigned* s, unsigned* m) {
    if (j < 4u) { *s = sc[j] & 0x3Fu; *m = sc[j + 4u] & 0x3Fu; }
    else { *s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6) << 4); *m = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4); }
}

// ============================ kernels ============================

// Q4_K fused dequant + canonical dot (one thread per output row). Proven.
__device__ __forceinline__ float q4k_row(const float* x, const unsigned char* wrow, unsigned K, unsigned nsuper) {
    float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
    float chunk_sums[64]; unsigned nch = 0u; unsigned spc = 8192u / 256u; float buf[256];
    for (unsigned b = 0u; b < nsuper; b++) {
        unsigned base = b * 256u; if (base >= K) break;
        const unsigned char* blk = wrow + b * 144u;
        float d = f16_to_f32((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
        float dmin = f16_to_f32((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
        const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
        unsigned is = 0u, qo = 0u, y = 0u;
        for (unsigned sub = 0u; sub < 4u; sub++) {
            unsigned s1, m1, s2, m2;
            get_scale_min_k4(is, sc, &s1, &m1);
            get_scale_min_k4(is + 1u, sc, &s2, &m2);
            float d1 = d * (float)s1, m1f = dmin * (float)m1, d2 = d * (float)s2, m2f = dmin * (float)m2;
            for (unsigned t = 0u; t < 32u; t++) { unsigned char q = qs[qo + t]; buf[y + t] = d1 * (float)(q & 0x0Fu) - m1f; buf[y + 32u + t] = d2 * (float)(q >> 4) - m2f; }
            y += 64u; qo += 32u; is += 2u;
        }
        unsigned avail = (256u < K - base) ? 256u : (K - base);
        unsigned full = avail - (avail % 8u);
        for (unsigned i = 0u; i < full; i += 8u) for (unsigned j = 0u; j < 8u; j++) { float p = x[base + i + j] * buf[i + j]; lanes[j] += p; }
        for (unsigned t = full; t < avail; t++) { float p = x[base + t] * buf[t]; lanes[t % 8u] += p; }
        bool last = (base + avail >= K);
        if (((b + 1u) % spc == 0u) || last) { float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j]; chunk_sums[nch++] = fixed_tree(l, 8u); for (int j = 0; j < 8; j++) lanes[j] = 0.0f; }
    }
    return fixed_tree(chunk_sums, nch);
}
__global__ void k_q4k(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = q4k_row(x, w + (size_t)gid * nsuper * 144u, K, nsuper);
}
__global__ void k_q4k_acc(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = out[gid] + q4k_row(x, w + (size_t)gid * nsuper * 144u, K, nsuper); // fuse residual
}

// Cooperative Q4_K linear: 8 threads per output row (one per reduction lane). Each lane
// thread sums ITS lane (positions p ≡ lane mod 8) in increasing-index order; the 8 lane
// sums are combined through shared memory in the exact fixed tree (per 8192-element chunk,
// then across chunks). Same arithmetic as k_q4k — identical lane order, identical tree —
// so bit-identical, but with coalesced x/W reads and 8× row parallelism instead of one
// serial thread per row. acc != 0 fuses the residual (out += dot).
__global__ void k_q4k_coop(const float* x, const unsigned char* w, float* out,
                           unsigned K, unsigned M, unsigned nsuper, int acc) {
    __shared__ float sh[8][8][8];                 // [row_in_block][lane][chunk]; chunks<=8 (nsuper<=256)
    unsigned lane = threadIdx.x & 7u;             // reduction lane 0..7
    unsigned rrow = threadIdx.x >> 3;             // row within block 0..7
    unsigned row  = blockIdx.x * 8u + rrow;
    unsigned nch_total = (nsuper + 31u) / 32u;
    if (row < M) {
        const unsigned char* wrow = w + (size_t)row * nsuper * 144u;
        float laneacc = 0.0f; unsigned sb_in_chunk = 0u, nch = 0u;
        for (unsigned b = 0u; b < nsuper; b++) {
            const unsigned char* blk = wrow + b * 144u;
            float d    = f16_to_f32((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
            float dmin = f16_to_f32((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
            const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
            unsigned gbase = b * 256u;
            for (unsigned tt = 0u; tt < 32u; tt++) {           // this lane's 32 positions, in order
                unsigned p = lane + tt * 8u;
                unsigned sub = p >> 6, within = p & 63u, t = within & 31u, half = within >> 5;
                unsigned s, m; get_scale_min_k4(sub * 2u + half, sc, &s, &m);
                unsigned q = qs[sub * 32u + t];
                unsigned nib = (half == 0u) ? (q & 0x0Fu) : (q >> 4);
                float d1 = d * (float)s, mf = dmin * (float)m;
                float val = d1 * (float)nib - mf;              // --fmad=false: separate rounding
                float pr = x[gbase + p] * val;
                laneacc += pr;
            }
            sb_in_chunk++;
            if (sb_in_chunk == 32u || b == nsuper - 1u) { sh[rrow][lane][nch] = laneacc; nch++; laneacc = 0.0f; sb_in_chunk = 0u; }
        }
    }
    __syncthreads();
    if (row < M && lane == 0u) {
        float cs[8];
        for (unsigned c = 0u; c < nch_total; c++) {
            float l[8]; for (int j = 0; j < 8; j++) l[j] = sh[rrow][j][c];
            cs[c] = fixed_tree(l, 8u);
        }
        float r = fixed_tree(cs, nch_total);
        out[row] = acc ? (out[row] + r) : r;
    }
}

// Q6_K integer linear (quantize x inline per super-block, integer dot). Proven.
__device__ __forceinline__ float q6k_row(const float* x, const unsigned char* wrow, unsigned nsuper) {
    float parts[256];   // one partial per super-block; sized for hidden up to 65536 (32B/70B/405B)
    for (unsigned s = 0u; s < nsuper; s++) {
        const float* xb = x + s * 256u;
        float amax = 0.0f; for (unsigned i = 0u; i < 256u; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        float dx = div_sw(amax, 127.0f);
        float inv = (dx != 0.0f) ? div_sw(1.0f, dx) : 0.0f;
        signed char qs[256];
        for (unsigned i = 0u; i < 256u; i++) { float sc = xb[i] * inv; float bias = (sc >= 0.0f) ? 0.5f : -0.5f; int q = (int)(sc + bias); q = q < -127 ? -127 : (q > 127 ? 127 : q); qs[i] = (signed char)q; }
        const unsigned char* blk = wrow + s * 210u;
        const unsigned char* ql = blk; const unsigned char* qh = blk + 128; const unsigned char* scb = blk + 192;
        float d_w = f16_to_f32((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
        int sums[16]; for (int j = 0; j < 16; j++) sums[j] = 0;
        for (unsigned hb = 0u; hb < 2u; hb++) {
            unsigned ql_p = hb * 64u, qh_p = hb * 32u, sc_p = hb * 8u, y = hb * 128u;
            for (unsigned pass = 0u; pass < 2u; pass++) {
                unsigned is = pass, l0 = pass * 16u; int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
                for (unsigned l = l0; l < l0 + 16u; l++) {
                    unsigned h = (unsigned)qh[qh_p + l], b0 = (unsigned)ql[ql_p + l], b1 = (unsigned)ql[ql_p + l + 32u];
                    a1 += ((int)((b0 & 0x0Fu) | ((h & 0x03u) << 4)) - 32) * (int)qs[y + l];
                    a2 += ((int)((b1 & 0x0Fu) | (((h >> 2) & 0x03u) << 4)) - 32) * (int)qs[y + l + 32u];
                    a3 += ((int)((b0 >> 4) | (((h >> 4) & 0x03u) << 4)) - 32) * (int)qs[y + l + 64u];
                    a4 += ((int)((b1 >> 4) | (((h >> 6) & 0x03u) << 4)) - 32) * (int)qs[y + l + 96u];
                }
                sums[sc_p + is] += a1; sums[sc_p + is + 2u] += a2; sums[sc_p + is + 4u] += a3; sums[sc_p + is + 6u] += a4;
            }
        }
        int acc = 0; for (int j = 0; j < 16; j++) acc += (int)((signed char)scb[j]) * sums[j];
        parts[s] = d_w * dx * (float)acc;
    }
    return fixed_tree(parts, nsuper);
}
__global__ void k_q6k(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = q6k_row(x, w + (size_t)gid * nsuper * 210u, nsuper);
}
__global__ void k_q6k_acc(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = out[gid] + q6k_row(x, w + (size_t)gid * nsuper * 210u, nsuper); // fuse residual
}

// Q6_K quantize-once: the activation x is identical for every output row of a linear,
// so quantize it ONCE (k_q6k_quantize) and let every row reuse it (k_q6k_dot), instead
// of re-quantizing x per row. Bit-identical to q6k_row (same amax/div_sw/clamp). This is
// the dominant speedup for wide layers (ffn_down, lm_head): on a 152k-row lm_head it
// removes ~152063/152064 of the quantization work.
__global__ void k_q6k_quantize(const float* x, float* out_dx, signed char* out_qs, unsigned nsuper) {
    unsigned s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= nsuper) return;
    const float* xb = x + s * 256u;
    float amax = 0.0f; for (unsigned i = 0u; i < 256u; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
    float dx = div_sw(amax, 127.0f);
    float inv = (dx != 0.0f) ? div_sw(1.0f, dx) : 0.0f;
    out_dx[s] = dx;
    signed char* qo = out_qs + s * 256u;
    for (unsigned i = 0u; i < 256u; i++) { float sc = xb[i] * inv; float bias = (sc >= 0.0f) ? 0.5f : -0.5f; int q = (int)(sc + bias); q = q < -127 ? -127 : (q > 127 ? 127 : q); qo[i] = (signed char)q; }
}
__device__ __forceinline__ float q6k_dot_pre(const float* xdx, const signed char* xqs, const unsigned char* wrow, unsigned nsuper) {
    float parts[256];
    for (unsigned s = 0u; s < nsuper; s++) {
        const signed char* qs = xqs + s * 256u;
        float dx = xdx[s];
        const unsigned char* blk = wrow + s * 210u;
        const unsigned char* ql = blk; const unsigned char* qh = blk + 128; const unsigned char* scb = blk + 192;
        float d_w = f16_to_f32((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
        int sums[16]; for (int j = 0; j < 16; j++) sums[j] = 0;
        for (unsigned hb = 0u; hb < 2u; hb++) {
            unsigned ql_p = hb * 64u, qh_p = hb * 32u, sc_p = hb * 8u, y = hb * 128u;
            for (unsigned pass = 0u; pass < 2u; pass++) {
                unsigned is = pass, l0 = pass * 16u; int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
                for (unsigned l = l0; l < l0 + 16u; l++) {
                    unsigned h = (unsigned)qh[qh_p + l], b0 = (unsigned)ql[ql_p + l], b1 = (unsigned)ql[ql_p + l + 32u];
                    a1 += ((int)((b0 & 0x0Fu) | ((h & 0x03u) << 4)) - 32) * (int)qs[y + l];
                    a2 += ((int)((b1 & 0x0Fu) | (((h >> 2) & 0x03u) << 4)) - 32) * (int)qs[y + l + 32u];
                    a3 += ((int)((b0 >> 4) | (((h >> 4) & 0x03u) << 4)) - 32) * (int)qs[y + l + 64u];
                    a4 += ((int)((b1 >> 4) | (((h >> 6) & 0x03u) << 4)) - 32) * (int)qs[y + l + 96u];
                }
                sums[sc_p + is] += a1; sums[sc_p + is + 2u] += a2; sums[sc_p + is + 4u] += a3; sums[sc_p + is + 6u] += a4;
            }
        }
        int acc = 0; for (int j = 0; j < 16; j++) acc += (int)((signed char)scb[j]) * sums[j];
        parts[s] = d_w * dx * (float)acc;
    }
    return fixed_tree(parts, nsuper);
}
__global__ void k_q6k_dot(const float* xdx, const signed char* xqs, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = q6k_dot_pre(xdx, xqs, w + (size_t)gid * nsuper * 210u, nsuper);
}
__global__ void k_q6k_dot_acc(const float* xdx, const signed char* xqs, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = out[gid] + q6k_dot_pre(xdx, xqs, w + (size_t)gid * nsuper * 210u, nsuper);
}

// Q8_0 linear: dequantize each row on the fly (scale * int8, pure mul) and run the
// SAME canonical dot (8 lanes by i%8, fixed_tree per 8192-chunk, fixed_tree over
// chunks). Matches dequantize_q8_0 + canonical_dot on the CPU. 34 bytes / 32 elems.
__device__ __forceinline__ float q8_0_row(const float* x, const unsigned char* wrow, unsigned K) {
    float chunk_sums[64]; unsigned nch = 0u; unsigned CHUNK = 8192u; unsigned done = 0u;
    while (done < K) {
        unsigned e = (done + CHUNK < K) ? (done + CHUNK) : K;
        unsigned n = e - done;
        float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
        unsigned full = n - (n % 8u), i = 0u;
        for (; i < full; i += 8u) for (unsigned j = 0u; j < 8u; j++) {
            unsigned idx = done + i + j, blk = idx >> 5, within = idx & 31u;
            const unsigned char* bp = wrow + blk * 34u;
            float d = f16_to_f32((unsigned short)((unsigned)bp[0] | ((unsigned)bp[1] << 8)));
            float wv = d * (float)((signed char)bp[2u + within]);
            float p = x[idx] * wv; lanes[j] += p;
        }
        for (; i < n; i++) {
            unsigned idx = done + i, blk = idx >> 5, within = idx & 31u;
            const unsigned char* bp = wrow + blk * 34u;
            float d = f16_to_f32((unsigned short)((unsigned)bp[0] | ((unsigned)bp[1] << 8)));
            float wv = d * (float)((signed char)bp[2u + within]);
            float p = x[idx] * wv; lanes[i % 8u] += p;
        }
        float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
        chunk_sums[nch++] = fixed_tree(l, 8u);
        done = e;
    }
    return fixed_tree(chunk_sums, nch);
}
__global__ void k_q8_0(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = q8_0_row(x, w + (size_t)gid * (K >> 5) * 34u, K);
}
__global__ void k_q8_0_acc(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = out[gid] + q8_0_row(x, w + (size_t)gid * (K >> 5) * 34u, K);
}

// Q5_0 linear: dequantize each row on the fly (scale * (5-bit - 16), pure mul) and
// run the SAME canonical dot. Matches dequantize_q5_0 + canonical_dot. 22 bytes / 32.
__device__ __forceinline__ float q5_0_val(const unsigned char* wrow, unsigned idx) {
    unsigned blk = idx >> 5, within = idx & 31u;
    const unsigned char* bp = wrow + blk * 22u;
    float scale = f16_to_f32((unsigned short)((unsigned)bp[0] | ((unsigned)bp[1] << 8)));
    unsigned qh = (unsigned)bp[2] | ((unsigned)bp[3] << 8) | ((unsigned)bp[4] << 16) | ((unsigned)bp[5] << 24);
    const unsigned char* qs = bp + 6;
    int v;
    if (within < 16u) { unsigned j = within; unsigned xh = ((qh >> j) << 4) & 0x10u; v = (int)((qs[j] & 0x0Fu) | xh) - 16; }
    else { unsigned j = within - 16u; unsigned xh = (qh >> (j + 12u)) & 0x10u; v = (int)((qs[j] >> 4) | xh) - 16; }
    return scale * (float)v;
}
__device__ __forceinline__ float q5_0_row(const float* x, const unsigned char* wrow, unsigned K) {
    float chunk_sums[64]; unsigned nch = 0u; unsigned CHUNK = 8192u; unsigned done = 0u;
    while (done < K) {
        unsigned e = (done + CHUNK < K) ? (done + CHUNK) : K;
        unsigned n = e - done;
        float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
        unsigned full = n - (n % 8u), i = 0u;
        for (; i < full; i += 8u) for (unsigned j = 0u; j < 8u; j++) { unsigned idx = done + i + j; float p = x[idx] * q5_0_val(wrow, idx); lanes[j] += p; }
        for (; i < n; i++) { unsigned idx = done + i; float p = x[idx] * q5_0_val(wrow, idx); lanes[i % 8u] += p; }
        float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
        chunk_sums[nch++] = fixed_tree(l, 8u);
        done = e;
    }
    return fixed_tree(chunk_sums, nch);
}
__global__ void k_q5_0(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = q5_0_row(x, w + (size_t)gid * (K >> 5) * 22u, K);
}
__global__ void k_q5_0_acc(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    out[gid] = out[gid] + q5_0_row(x, w + (size_t)gid * (K >> 5) * 22u, K);
}

// Dequantize one Q4_K row to f32 (the embedding lookup). One thread per super-block.
__global__ void k_q4k_dequant(const unsigned char* w, float* out, unsigned nsuper) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= nsuper) return;
    const unsigned char* blk = w + b * 144u;
    float d = f16_to_f32((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
    float dmin = f16_to_f32((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    float* obuf = out + b * 256u;
    unsigned is = 0u, qo = 0u, y = 0u;
    for (unsigned sub = 0u; sub < 4u; sub++) {
        unsigned s1, m1, s2, m2;
        get_scale_min_k4(is, sc, &s1, &m1);
        get_scale_min_k4(is + 1u, sc, &s2, &m2);
        float d1 = d * (float)s1, m1f = dmin * (float)m1, d2 = d * (float)s2, m2f = dmin * (float)m2;
        for (unsigned t = 0u; t < 32u; t++) { unsigned char q = qs[qo + t]; obuf[y + t] = d1 * (float)(q & 0x0Fu) - m1f; obuf[y + 32u + t] = d2 * (float)(q >> 4) - m2f; }
        y += 64u; qo += 32u; is += 2u;
    }
}

// rms_norm (regime-2). Proven. One thread per row; feat < 8192 so one chunk is exact.
__global__ void k_rms(const float* x, const float* wt, float* out, unsigned feat, unsigned rows, float eps) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= rows) return;
    const float* row = x + (size_t)gid * feat;
    float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
    unsigned full = feat - (feat % 8u), i = 0u;
    for (; i < full; i += 8u) for (unsigned j = 0u; j < 8u; j++) { float p = row[i + j] * row[i + j]; lanes[j] += p; }
    for (; i < feat; i++) { unsigned j = i % 8u; float p = row[i] * row[i]; lanes[j] += p; }
    float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
    float sumsq = fixed_tree(l, 8u);
    float mean = div_sw(sumsq, (float)feat);
    float scale = div_sw(1.0f, sqrtf(mean + eps));
    float* orow = out + (size_t)gid * feat;
    for (unsigned k = 0u; k < feat; k++) { float t = row[k] * scale; orow[k] = t * wt[k]; }
}

// Multi-head attention (GQA + KV cache), one thread per query head. Proven.
__global__ void k_attention(const float* q, const float* kc, const float* vc, float* xbout,
                            unsigned n_heads, unsigned head_size, unsigned kv_dim, unsigned kv_mul, unsigned pos) {
    unsigned h = blockIdx.x * blockDim.x + threadIdx.x; if (h >= n_heads) return;
    float scores[4096];
    unsigned qoff = h * head_size, kvhead = h / kv_mul; float sqrt_hs = sqrtf((float)head_size);
    unsigned tlen = pos + 1u;
    for (unsigned t = 0u; t < tlen; t++) {
        unsigned koff = t * kv_dim + kvhead * head_size;
        float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
        unsigned full = head_size - (head_size % 8u), dd = 0u;
        for (; dd < full; dd += 8u) for (unsigned j = 0u; j < 8u; j++) { float p = q[qoff + dd + j] * kc[koff + dd + j]; lanes[j] += p; }
        for (; dd < head_size; dd++) { unsigned j = dd % 8u; float p = q[qoff + dd] * kc[koff + dd]; lanes[j] += p; }
        float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
        scores[t] = div_sw(fixed_tree(l, 8u), sqrt_hs);
    }
    float mx = -INFINITY; for (unsigned t = 0u; t < tlen; t++) { if (scores[t] > mx) mx = scores[t]; }
    for (unsigned t = 0u; t < tlen; t++) scores[t] = vt_expf(scores[t] - mx);
    {
        float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
        unsigned full = tlen - (tlen % 8u), t = 0u;
        for (; t < full; t += 8u) for (unsigned j = 0u; j < 8u; j++) lanes[j] += scores[t + j];
        for (; t < tlen; t++) lanes[t % 8u] += scores[t];
        float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
        float inv = div_sw(1.0f, fixed_tree(l, 8u));
        for (unsigned k = 0u; k < tlen; k++) scores[k] = scores[k] * inv;
    }
    for (unsigned dd = 0u; dd < head_size; dd++) {
        float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
        unsigned full = tlen - (tlen % 8u), t = 0u;
        for (; t < full; t += 8u) for (unsigned j = 0u; j < 8u; j++) { unsigned voff = (t + j) * kv_dim + kvhead * head_size; float p = scores[t + j] * vc[voff + dd]; lanes[j] += p; }
        for (; t < tlen; t++) { unsigned voff = t * kv_dim + kvhead * head_size; float p = scores[t] * vc[voff + dd]; lanes[t % 8u] += p; }
        float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
        xbout[qoff + dd] = fixed_tree(l, 8u);
    }
}

// SwiGLU inner: silu(gate) * up.
__global__ void k_silu_mul(const float* gate, const float* up, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float g = gate[i];
    out[i] = div_sw(g, 1.0f + vt_expf(-g)) * up[i];
}

// RoPE apply, interleaved (Llama/Mistral): rotate adjacent pairs (2i, 2i+1).
__global__ void k_rope(const float* x, const float* cosc, const float* sinc, float* out,
                       unsigned seq, unsigned n_heads, unsigned head_dim) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= seq * n_heads) return;
    unsigned s = gid / n_heads; unsigned hd = head_dim / 2u; unsigned head_off = gid * head_dim;
    for (unsigned i = 0u; i < hd; i++) {
        float a = x[head_off + 2u * i], b = x[head_off + 2u * i + 1u];
        float c = cosc[s * hd + i], sn = sinc[s * hd + i];
        float p1 = a * c, p2 = b * sn; out[head_off + 2u * i] = p1 - p2;       // --fmad=false: separate rounding
        float p3 = a * sn, p4 = b * c; out[head_off + 2u * i + 1u] = p3 + p4;
    }
}

// RoPE apply, NeoX (Qwen2): rotate split-halves (j, j + head_dim/2). Same cos/sin
// cache as interleaved; only the pairing differs.
__global__ void k_rope_neox(const float* x, const float* cosc, const float* sinc, float* out,
                            unsigned seq, unsigned n_heads, unsigned head_dim) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= seq * n_heads) return;
    unsigned s = gid / n_heads; unsigned hd = head_dim / 2u; unsigned head_off = gid * head_dim;
    for (unsigned j = 0u; j < hd; j++) {
        float a = x[head_off + j], b = x[head_off + j + hd];
        float c = cosc[s * hd + j], sn = sinc[s * hd + j];
        float p1 = a * c, p2 = b * sn; out[head_off + j] = p1 - p2;
        float p3 = a * sn, p4 = b * c; out[head_off + j + hd] = p3 + p4;
    }
}

// Optional QKV bias (Qwen2): v[i] += b[i], in place, before rope/cache.
__global__ void k_add_bias(float* v, const float* b, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    v[i] = v[i] + b[i];
}

// ====================================================================
// TIER-2 EXPERIMENT (regime-3): integer-activation Q4_K linear.
// Quantize x to int8 per 256-block (bsums per 32-group), then the per-block dot
// is EXACT INTEGER arithmetic (Σ int8·int8 in int32) — associative, so it is
// order-independent and identical on any hardware, and can use the int8 dot
// instruction __dp4a / int8 tensor cores. Mirrors the CPU's dot_q4k_q8_block +
// quantize_block_q8 bit-for-bit. This is a DIFFERENT numeric regime from the
// shipped float Q4_K (a re-anchor), kept behind test entry points only.
// ====================================================================
__global__ void k_q4k_q8_quantize(const float* x, float* out_dx, signed char* out_qs, int* out_bsums, unsigned nsuper) {
    unsigned s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= nsuper) return;
    const float* xb = x + s * 256u;
    float amax = 0.0f; for (unsigned i = 0u; i < 256u; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
    float d = amax / 127.0f;
    float inv = (d != 0.0f) ? 1.0f / d : 0.0f;
    out_dx[s] = d;
    signed char* qo = out_qs + s * 256u;
    int* bs = out_bsums + s * 8u; for (int g = 0; g < 8; g++) bs[g] = 0;
    for (unsigned i = 0u; i < 256u; i++) {
        float scaled = xb[i] * inv; float bias = (scaled >= 0.0f) ? 0.5f : -0.5f;
        int q = (int)(scaled + bias); q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qo[i] = (signed char)q; bs[i / 32u] += q;
    }
}
// One Q4_K super-block × a Q8 x-block, exact-integer. mode: 0 = plain loop, 1 = __dp4a.
__device__ __forceinline__ float q4k_q8_part(float d_x, const signed char* xq, const int* bsums,
                                             const unsigned char* blk, int mode) {
    float d = f16_to_f32((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
    float dmin = f16_to_f32((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    int acc = 0, acc_m = 0;
    for (unsigned half = 0u; half < 4u; half++) {
        unsigned s1, m1, s2, m2;
        get_scale_min_k4(half * 2u, sc, &s1, &m1);
        get_scale_min_k4(half * 2u + 1u, sc, &s2, &m2);
        unsigned q_off = half * 32u, base = half * 64u;
        int s_lo = 0, s_hi = 0;
        if (mode == 1) {                              // __dp4a: 4 int8 products/instr (exact integer)
            for (unsigned k = 0u; k < 32u; k += 4u) {
                int nlo = (int)((unsigned)(qs[q_off+k]&0x0F) | ((unsigned)(qs[q_off+k+1]&0x0F)<<8) | ((unsigned)(qs[q_off+k+2]&0x0F)<<16) | ((unsigned)(qs[q_off+k+3]&0x0F)<<24));
                int nhi = (int)((unsigned)(qs[q_off+k]>>4) | ((unsigned)(qs[q_off+k+1]>>4)<<8) | ((unsigned)(qs[q_off+k+2]>>4)<<16) | ((unsigned)(qs[q_off+k+3]>>4)<<24));
                int xlo = (int)((unsigned)(unsigned char)xq[base+k] | ((unsigned)(unsigned char)xq[base+k+1]<<8) | ((unsigned)(unsigned char)xq[base+k+2]<<16) | ((unsigned)(unsigned char)xq[base+k+3]<<24));
                int xhi = (int)((unsigned)(unsigned char)xq[base+32+k] | ((unsigned)(unsigned char)xq[base+32+k+1]<<8) | ((unsigned)(unsigned char)xq[base+32+k+2]<<16) | ((unsigned)(unsigned char)xq[base+32+k+3]<<24));
                s_lo = __dp4a(nlo, xlo, s_lo);
                s_hi = __dp4a(nhi, xhi, s_hi);
            }
        } else {                                      // plain integer loop (reference order)
            for (unsigned l = 0u; l < 32u; l++) {
                s_lo += (int)(qs[q_off + l] & 0x0F) * (int)xq[base + l];
                s_hi += (int)(qs[q_off + l] >> 4)   * (int)xq[base + 32u + l];
            }
        }
        acc   += (int)s1 * s_lo + (int)s2 * s_hi;
        acc_m += (int)m1 * bsums[half * 2u] + (int)m2 * bsums[half * 2u + 1u];
    }
    float a1 = d * d_x, a2 = dmin * d_x;              // d*xq.d*acc - dmin*xq.d*acc_m  (--fmad=false)
    return a1 * (float)acc - a2 * (float)acc_m;
}
__global__ void k_q4k_q8_dot(const float* dx, const signed char* qs, const int* bsums,
                             const unsigned char* w, float* out, unsigned M, unsigned nsuper, int mode) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    const unsigned char* wrow = w + (size_t)gid * nsuper * 144u;
    float parts[256];
    for (unsigned s = 0u; s < nsuper; s++)
        parts[s] = q4k_q8_part(dx[s], qs + s * 256u, bsums + s * 8u, wrow + s * 144u, mode);
    out[gid] = fixed_tree(parts, nsuper);
}
// Warp-per-row integer Q4_K: 32 threads cooperate on ONE row, each computing some
// super-blocks' parts (reads the row's weights COALESCED), then the per-block floats
// are combined by thread 0 in the fixed canonical order. The per-block dot is exact
// integer, so splitting blocks across the warp doesn't change a bit. inner mode:
// 0 = plain, 1 = __dp4a.
__global__ void k_q4k_q8_dot_warp(const float* dx, const signed char* qs, const int* bsums,
                                  const unsigned char* w, float* out, unsigned M, unsigned nsuper, int mode) {
    unsigned row = blockIdx.x; if (row >= M) return;
    unsigned t = threadIdx.x;                      // 0..31 (one warp = one row)
    __shared__ float parts[256];
    const unsigned char* wrow = w + (size_t)row * nsuper * 144u;
    for (unsigned s = t; s < nsuper; s += 32u)
        parts[s] = q4k_q8_part(dx[s], qs + s * 256u, bsums + s * 8u, wrow + s * 144u, mode);
    __syncthreads();
    if (t == 0u) out[row] = fixed_tree(parts, nsuper);
}

// ====================================================================
// regime-3 forward kernels: integer activation + RUNNING-SUM combine across
// super-blocks (matches the engine's linear_q4_k_integer_rs / linear_q6_k_integer_rs).
// One thread per output row (the per-token forward is a matvec, N=1). Q4_K reuses
// q4k_q8_part; Q6_K's per-super-block integer dot is inlined here. acc fuses the residual.
// ====================================================================
__device__ __forceinline__ float q6k_q8_part(const float* xdx, const signed char* xqs, const unsigned char* blk, unsigned s) {
    const signed char* qs = xqs + s * 256u;
    float dx = xdx[s];
    const unsigned char* ql = blk; const unsigned char* qh = blk + 128; const unsigned char* scb = blk + 192;
    float d_w = f16_to_f32((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
    int sums[16]; for (int j = 0; j < 16; j++) sums[j] = 0;
    for (unsigned hb = 0u; hb < 2u; hb++) {
        unsigned ql_p = hb * 64u, qh_p = hb * 32u, sc_p = hb * 8u, y = hb * 128u;
        for (unsigned pass = 0u; pass < 2u; pass++) {
            unsigned is = pass, l0 = pass * 16u; int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
            for (unsigned l = l0; l < l0 + 16u; l++) {
                unsigned h = (unsigned)qh[qh_p + l], b0 = (unsigned)ql[ql_p + l], b1 = (unsigned)ql[ql_p + l + 32u];
                a1 += ((int)((b0 & 0x0Fu) | ((h & 0x03u) << 4)) - 32) * (int)qs[y + l];
                a2 += ((int)((b1 & 0x0Fu) | (((h >> 2) & 0x03u) << 4)) - 32) * (int)qs[y + l + 32u];
                a3 += ((int)((b0 >> 4) | (((h >> 4) & 0x03u) << 4)) - 32) * (int)qs[y + l + 64u];
                a4 += ((int)((b1 >> 4) | (((h >> 6) & 0x03u) << 4)) - 32) * (int)qs[y + l + 96u];
            }
            sums[sc_p + is] += a1; sums[sc_p + is + 2u] += a2; sums[sc_p + is + 4u] += a3; sums[sc_p + is + 6u] += a4;
        }
    }
    int acc = 0; for (int j = 0; j < 16; j++) acc += (int)((signed char)scb[j]) * sums[j];
    return d_w * dx * (float)acc;
}
__global__ void k_q4k_rs(const float* dx, const signed char* qs, const int* bsums, const unsigned char* w, float* out, unsigned M, unsigned nsuper, int acc) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    const unsigned char* wrow = w + (size_t)gid * nsuper * 144u;
    float sum = 0.0f;
    for (unsigned s = 0u; s < nsuper; s++) sum += q4k_q8_part(dx[s], qs + s * 256u, bsums + s * 8u, wrow + s * 144u, 1);
    out[gid] = acc ? (out[gid] + sum) : sum;
}
__global__ void k_q6k_rs(const float* dx, const signed char* qs, const unsigned char* w, float* out, unsigned M, unsigned nsuper, int acc) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    const unsigned char* wrow = w + (size_t)gid * nsuper * 210u;
    float sum = 0.0f;
    for (unsigned s = 0u; s < nsuper; s++) sum += q6k_q8_part(dx, qs, wrow + s * 210u, s);
    out[gid] = acc ? (out[gid] + sum) : sum;
}

// ====================================================================
// Batched prefill GEMMs (regime-3): out[N tokens][M rows] (token-major) = running-sum integer
// dot, register-tiled (ported from the v4 dp4a kernel) so weights are staged once in shared and
// reused across the token batch. acc != 0 fuses the residual. Bit-identical to the per-token
// k_q4k_rs / k_q6k_rs looped over tokens (same integer dot, same running-sum order).
// ====================================================================
// Batched Q8 quantize: x[N][K] -> dx[N][ns], qs[N][ns*256] int8, bsums[N][ns*8].
__global__ void k_q8_quant_batch(const float* x, float* dx, signed char* qs, int* bsums, unsigned N, unsigned K, unsigned ns) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N * ns) return;
    unsigned n = idx / ns, s = idx % ns;
    const float* xb = x + (size_t)n * K + s * 256u;
    float amax = 0.0f; for (int i = 0; i < 256; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
    float d = amax / 127.0f, inv = d != 0.0f ? 1.0f / d : 0.0f;
    dx[(size_t)n * ns + s] = d;
    signed char* qo = qs + ((size_t)n * ns + s) * 256u;
    int* b = bsums + ((size_t)n * ns + s) * 8u; for (int g = 0; g < 8; g++) b[g] = 0;
    for (int i = 0; i < 256; i++) { float sc = xb[i] * inv; float bias = sc >= 0.0f ? 0.5f : -0.5f; int q = (int)(sc + bias); q = q < -127 ? -127 : (q > 127 ? 127 : q); qo[i] = (signed char)q; b[i / 32] += q; }
}
#define PBM 64
#define PBN 64
// Q4_K batched GEMM. W[M][ns*144], xq[N][ns*256], out[N][M] token-major. acc: out[n][m]+=.
__global__ void __launch_bounds__(256) k_q4k_rs_gemm(const unsigned char* W, const float* dx, const signed char* xq, const int* bsums,
                                                     float* out, unsigned M, unsigned N, unsigned ns, int acc) {
    unsigned tm = blockIdx.x * PBM, tn = blockIdx.y * PBN, tid = threadIdx.x, trow = tid >> 4, tcol = tid & 15u;
    __shared__ signed char wsh[PBM][256];
    __shared__ signed char xsh[PBN][256];
    __shared__ unsigned char scsh[PBM][8], msh[PBM][8];
    __shared__ float dxsh[PBN];
    __shared__ int bsh[PBN][8];
    float outr[4][4]; for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) outr[a][b] = 0.0f;
    float drow[4], dmrow[4];
    for (unsigned s = 0u; s < ns; s++) {
        for (unsigned i = tid; i < PBM * 256u; i += 256u) {
            unsigned r = i >> 8, e = i & 255u, wr = tm + r; if (wr >= M) { wsh[r][e] = 0; continue; }
            const unsigned char* qs = W + (size_t)wr * ns * 144u + s * 144u + 16u;
            unsigned half = e >> 6, within = e & 63u, lo = within < 32u, l = within & 31u, q = qs[half * 32u + l];
            wsh[r][e] = (signed char)(lo ? (q & 0x0Fu) : (q >> 4));
        }
        for (unsigned i = tid; i < PBN * 256u; i += 256u) { unsigned c = i >> 8, e = i & 255u, tc = tn + c; xsh[c][e] = (tc < N) ? xq[((size_t)tc * ns + s) * 256u + e] : 0; }
        for (unsigned i = tid; i < PBM * 8u; i += 256u) { unsigned r = i >> 3, sub = i & 7u, wr = tm + r; if (wr >= M) { scsh[r][sub] = 0; msh[r][sub] = 0; continue; } const unsigned char* scb = W + (size_t)wr * ns * 144u + s * 144u + 4u; unsigned sv, mv; get_scale_min_k4(sub, scb, &sv, &mv); scsh[r][sub] = (unsigned char)sv; msh[r][sub] = (unsigned char)mv; }
        for (unsigned i = tid; i < PBN; i += 256u) { unsigned tc = tn + i; if (tc < N) { dxsh[i] = dx[(size_t)tc * ns + s]; for (int g = 0; g < 8; g++) bsh[i][g] = bsums[((size_t)tc * ns + s) * 8u + g]; } else { dxsh[i] = 0; for (int g = 0; g < 8; g++) bsh[i][g] = 0; } }
        for (int a = 0; a < 4; a++) { unsigned wr = tm + trow * 4u + a; if (wr < M) { const unsigned char* blk = W + (size_t)wr * ns * 144u + s * 144u; drow[a] = f16_to_f32(blk[0] | (blk[1] << 8)); dmrow[a] = f16_to_f32(blk[2] | (blk[3] << 8)); } }
        __syncthreads();
        for (int a = 0; a < 4; a++) {
            unsigned r = trow * 4u + a;
            for (int b = 0; b < 4; b++) {
                unsigned c = tcol * 4u + b; int accq = 0, accm = 0;
                for (unsigned half = 0u; half < 4u; half++) {
                    unsigned base = half * 64u; int slo = 0, shi = 0;
                    const signed char* wlo = &wsh[r][base]; const signed char* whi = &wsh[r][base + 32];
                    const signed char* xlo = &xsh[c][base]; const signed char* xhi = &xsh[c][base + 32];
                    for (int k = 0; k < 32; k += 4) { slo = __dp4a(*(const int*)(wlo + k), *(const int*)(xlo + k), slo); shi = __dp4a(*(const int*)(whi + k), *(const int*)(xhi + k), shi); }
                    accq += (int)scsh[r][half * 2u] * slo + (int)scsh[r][half * 2u + 1u] * shi;
                    accm += (int)msh[r][half * 2u] * bsh[c][half * 2u] + (int)msh[r][half * 2u + 1u] * bsh[c][half * 2u + 1u];
                }
                float a1 = drow[a] * dxsh[c], a2 = dmrow[a] * dxsh[c];
                outr[a][b] += a1 * (float)accq - a2 * (float)accm;   // running sum over super-blocks
            }
        }
        __syncthreads();
    }
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
        unsigned wr = tm + trow * 4u + a, tc = tn + tcol * 4u + b;
        if (wr < M && tc < N) { size_t o = (size_t)tc * M + wr; out[o] = acc ? (out[o] + outr[a][b]) : outr[a][b]; }
    }
}

// ============================ host driver (C ABI) ============================

typedef struct {
    int dim, hidden, n_layers, n_heads, head_size, kv_dim, kv_mul, vocab, max_seq;
    int rope_neox;               // 1 = NeoX split-half rotation (Qwen2); 0 = interleaved (Llama)
    int regime;                  // 2 = shipped (float Q4_K, fixed_tree Q6_K); 3 = integer + running-sum
    float rms_eps, rope_theta;
} VtCfg;

typedef struct VtCuda {
    VtCfg cfg;
    int q_out;
    unsigned char *d_wcls; int wcls_q6k;
    float *d_rms_final;
    float **d_rms_att, **d_rms_ffn;
    unsigned char **d_wq, **d_wk, **d_wv, **d_wo, **d_w1, **d_w2, **d_w3;
    int *q6k_wq, *q6k_wk, *q6k_wv, *q6k_wo, *q6k_w1, *q6k_w2, *q6k_w3;
    float **d_bq, **d_bk, **d_bv;    // optional QKV bias (null when absent)
    float **d_keyc, **d_valc;
    float *d_q8_dx; signed char *d_q8_qs;   // quantize-once scratch (x quantized per matmul)
    int *d_q8_bsums;                         // per-32-group sums of x int8 (Q4_K integer dmin term)
    int regime;                              // 2 = shipped float/fixed_tree; 3 = integer + running-sum
    float *d_x, *d_xb, *d_q, *d_att, *d_gate, *d_up, *d_inner, *d_logits, *d_cos, *d_sin;
    int err;
} VtCuda;

static unsigned char* dev_bytes(const unsigned char* src, size_t len, int* err) {
    unsigned char* d = nullptr;
    if (cudaMalloc(&d, len) != cudaSuccess) { *err = 1; return nullptr; }
    if (cudaMemcpy(d, src, len, cudaMemcpyHostToDevice) != cudaSuccess) { *err = 1; }
    return d;
}
static float* dev_floats(const float* src, size_t n, int* err) {
    float* d = nullptr;
    if (cudaMalloc(&d, n * 4) != cudaSuccess) { *err = 1; return nullptr; }
    if (cudaMemcpy(d, src, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) { *err = 1; }
    return d;
}
static float* dev_zeros(size_t n, int* err) {
    float* d = nullptr; size_t b = (n ? n : 1) * 4;
    if (cudaMalloc(&d, b) != cudaSuccess) { *err = 1; return nullptr; }
    if (cudaMemset(d, 0, b) != cudaSuccess) { *err = 1; }
    return d;
}

extern "C" VtCuda* vt_cuda_new(const VtCfg* cfg) {
    VtCuda* h = (VtCuda*)calloc(1, sizeof(VtCuda));
    h->cfg = *cfg;
    h->regime = cfg->regime ? cfg->regime : 2;
    h->q_out = cfg->n_heads * cfg->head_size;
    int L = cfg->n_layers;
    h->d_rms_att = (float**)calloc(L, sizeof(float*));
    h->d_rms_ffn = (float**)calloc(L, sizeof(float*));
    h->d_wq = (unsigned char**)calloc(L, sizeof(void*)); h->d_wk = (unsigned char**)calloc(L, sizeof(void*));
    h->d_wv = (unsigned char**)calloc(L, sizeof(void*)); h->d_wo = (unsigned char**)calloc(L, sizeof(void*));
    h->d_w1 = (unsigned char**)calloc(L, sizeof(void*)); h->d_w2 = (unsigned char**)calloc(L, sizeof(void*));
    h->d_w3 = (unsigned char**)calloc(L, sizeof(void*));
    h->q6k_wq = (int*)calloc(L, 4); h->q6k_wk = (int*)calloc(L, 4); h->q6k_wv = (int*)calloc(L, 4);
    h->q6k_wo = (int*)calloc(L, 4); h->q6k_w1 = (int*)calloc(L, 4); h->q6k_w2 = (int*)calloc(L, 4); h->q6k_w3 = (int*)calloc(L, 4);
    h->d_bq = (float**)calloc(L, sizeof(float*)); h->d_bk = (float**)calloc(L, sizeof(float*)); h->d_bv = (float**)calloc(L, sizeof(float*));
    h->d_keyc = (float**)calloc(L, sizeof(float*));
    h->d_valc = (float**)calloc(L, sizeof(float*));
    return h;
}

extern "C" void vt_cuda_set_wcls(VtCuda* h, const unsigned char* b, size_t len, int q6k) { h->d_wcls = dev_bytes(b, len, &h->err); h->wcls_q6k = q6k; }
extern "C" void vt_cuda_set_rms_final(VtCuda* h, const float* w) { h->d_rms_final = dev_floats(w, h->cfg.dim, &h->err); }

extern "C" void vt_cuda_set_layer(VtCuda* h, int l,
        const float* rms_att, const float* rms_ffn,
        const unsigned char* wq, size_t wq_len, int wq_q6k,
        const unsigned char* wk, size_t wk_len, int wk_q6k,
        const unsigned char* wv, size_t wv_len, int wv_q6k,
        const unsigned char* wo, size_t wo_len, int wo_q6k,
        const unsigned char* w1, size_t w1_len, int w1_q6k,
        const unsigned char* w2, size_t w2_len, int w2_q6k,
        const unsigned char* w3, size_t w3_len, int w3_q6k,
        const float* bq, const float* bk, const float* bv) {  // bias ptrs (null when absent)
    int dim = h->cfg.dim;
    h->d_rms_att[l] = dev_floats(rms_att, dim, &h->err);
    h->d_rms_ffn[l] = dev_floats(rms_ffn, dim, &h->err);
    if (bq) h->d_bq[l] = dev_floats(bq, h->q_out, &h->err);
    if (bk) h->d_bk[l] = dev_floats(bk, h->cfg.kv_dim, &h->err);
    if (bv) h->d_bv[l] = dev_floats(bv, h->cfg.kv_dim, &h->err);
    h->d_wq[l] = dev_bytes(wq, wq_len, &h->err); h->q6k_wq[l] = wq_q6k;
    h->d_wk[l] = dev_bytes(wk, wk_len, &h->err); h->q6k_wk[l] = wk_q6k;
    h->d_wv[l] = dev_bytes(wv, wv_len, &h->err); h->q6k_wv[l] = wv_q6k;
    h->d_wo[l] = dev_bytes(wo, wo_len, &h->err); h->q6k_wo[l] = wo_q6k;
    h->d_w1[l] = dev_bytes(w1, w1_len, &h->err); h->q6k_w1[l] = w1_q6k;
    h->d_w2[l] = dev_bytes(w2, w2_len, &h->err); h->q6k_w2[l] = w2_q6k;
    h->d_w3[l] = dev_bytes(w3, w3_len, &h->err); h->q6k_w3[l] = w3_q6k;
}

extern "C" int vt_cuda_finalize(VtCuda* h) {
    VtCfg* c = &h->cfg;
    int L = c->n_layers;
    size_t kv = (size_t)c->max_seq * c->kv_dim;
    for (int l = 0; l < L; l++) { h->d_keyc[l] = dev_zeros(kv, &h->err); h->d_valc[l] = dev_zeros(kv, &h->err); }
    h->d_x = dev_zeros(c->dim, &h->err); h->d_xb = dev_zeros(c->dim, &h->err);
    h->d_q = dev_zeros(h->q_out, &h->err); h->d_att = dev_zeros(h->q_out, &h->err);
    h->d_gate = dev_zeros(c->hidden, &h->err); h->d_up = dev_zeros(c->hidden, &h->err);
    h->d_inner = dev_zeros(c->hidden, &h->err); h->d_logits = dev_zeros(c->vocab, &h->err);
    h->d_cos = dev_zeros(c->head_size / 2, &h->err); h->d_sin = dev_zeros(c->head_size / 2, &h->err);
    // Q6_K quantize-once scratch: one dx + 256 int8 per super-block of the widest input.
    int maxfeat = c->dim > c->hidden ? c->dim : c->hidden;
    int maxnsuper = (maxfeat + 255) / 256;
    h->d_q8_dx = dev_zeros(maxnsuper, &h->err);
    if (cudaMalloc(&h->d_q8_qs, (size_t)maxnsuper * 256) != cudaSuccess) h->err = 1;
    if (cudaMalloc(&h->d_q8_bsums, (size_t)maxnsuper * 8 * 4) != cudaSuccess) h->err = 1;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) h->err = 1;
    return h->err;
}

// kind: 0 = Q4_K, 1 = Q6_K, 2 = Q8_0, 3 = Q5_0. q8dx/q8qs/q8bsums are quantize-once scratch.
// regime 3 routes Q4_K/Q6_K to the integer + running-sum kernels (matches the engine's
// regime-3); other dtypes are unchanged across regimes.
static inline void lin(const float* in, const unsigned char* w, int kind, float* out, int M, int K, int acc,
                       float* q8dx, signed char* q8qs, int regime, int* q8bsums) {
    int nsuper = K / 256;
    dim3 g((M + 63) / 64), b(64);
    if (regime == 3 && kind == 0) {            // Q4_K integer + running sum
        k_q4k_q8_quantize<<<(nsuper + 63) / 64, 64>>>(in, q8dx, q8qs, q8bsums, nsuper);
        k_q4k_rs<<<g, b>>>(q8dx, q8qs, q8bsums, w, out, M, nsuper, acc);
        return;
    }
    if (regime == 3 && kind == 1) {            // Q6_K integer + running sum
        k_q6k_quantize<<<(nsuper + 63) / 64, 64>>>(in, q8dx, q8qs, nsuper);
        k_q6k_rs<<<g, b>>>(q8dx, q8qs, w, out, M, nsuper, acc);
        return;
    }
    if (kind == 1) {
        if (q8dx) {  // quantize-once (fast): one quantize of the input, then per-row integer dot
            k_q6k_quantize<<<(nsuper + 63) / 64, 64>>>(in, q8dx, q8qs, nsuper);
            if (acc) k_q6k_dot_acc<<<g, b>>>(q8dx, q8qs, w, out, K, M, nsuper);
            else     k_q6k_dot<<<g, b>>>(q8dx, q8qs, w, out, K, M, nsuper);
        } else {     // fused fallback (re-quantizes per row)
            if (acc) k_q6k_acc<<<g, b>>>(in, w, out, K, M, nsuper); else k_q6k<<<g, b>>>(in, w, out, K, M, nsuper);
        }
    }
    else if (kind == 2) { if (acc) k_q8_0_acc<<<g, b>>>(in, w, out, K, M);        else k_q8_0<<<g, b>>>(in, w, out, K, M); }
    else if (kind == 3) { if (acc) k_q5_0_acc<<<g, b>>>(in, w, out, K, M);        else k_q5_0<<<g, b>>>(in, w, out, K, M); }
    else                { k_q4k_coop<<<(M + 7) / 8, 64>>>(in, w, out, K, M, nsuper, acc); }  // cooperative, coalesced
}

extern "C" void vt_cuda_reset(VtCuda* h) {
    VtCfg* c = &h->cfg;
    size_t kv = (size_t)c->max_seq * c->kv_dim * 4;
    for (int l = 0; l < c->n_layers; l++) { cudaMemset(h->d_keyc[l], 0, kv); cudaMemset(h->d_valc[l], 0, kv); }
}

// Logits for the embedding row `x_embed` (dim floats, dequantized on the host to
// match the CPU exactly) at position `pos`. cos/sin are the RoPE cache (head_size/2
// each) computed on the host with the SAME libm the CPU engine uses.
extern "C" int vt_cuda_step(VtCuda* h, const float* x_embed, unsigned pos, const float* cosc, const float* sinc, float* out_logits) {
    VtCfg* c = &h->cfg;
    int dim = c->dim, kv_dim = c->kv_dim, hidden = c->hidden, head_size = c->head_size, n_heads = c->n_heads;
    cudaMemcpy(h->d_cos, cosc, (head_size / 2) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_sin, sinc, (head_size / 2) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_x, x_embed, (size_t)dim * 4, cudaMemcpyHostToDevice);  // embedding lookup done on host
    int dbg = getenv("VITNI_DBG") != nullptr;
    auto l1 = [&](const char* tag, float* dptr, int n) {
        if (!dbg) return;
        cudaDeviceSynchronize();
        float* tmp = (float*)malloc((size_t)n * 4);
        cudaError_t e = cudaMemcpy(tmp, dptr, (size_t)n * 4, cudaMemcpyDeviceToHost);
        double s = 0; for (int i = 0; i < n; i++) s += fabs((double)tmp[i]);
        fprintf(stderr, "[dbg] %-16s L1=%.4f  lastErr=%s\n", tag, s, cudaGetErrorString(cudaGetLastError()));
        (void)e; free(tmp);
    };
    l1("embed d_x", h->d_x, dim);
    for (int l = 0; l < c->n_layers; l++) {
        k_rms<<<1, 64>>>(h->d_x, h->d_rms_att[l], h->d_xb, dim, 1, c->rms_eps);
        lin(h->d_xb, h->d_wq[l], h->q6k_wq[l], h->d_q, h->q_out, dim, 0, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);
        float* kptr = h->d_keyc[l] + (size_t)pos * kv_dim;
        float* vptr = h->d_valc[l] + (size_t)pos * kv_dim;
        lin(h->d_xb, h->d_wk[l], h->q6k_wk[l], kptr, kv_dim, dim, 0, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);
        lin(h->d_xb, h->d_wv[l], h->q6k_wv[l], vptr, kv_dim, dim, 0, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);
        // optional QKV bias (Qwen2), after projection and BEFORE rope/cache-use
        if (h->d_bq[l]) k_add_bias<<<(h->q_out + 255) / 256, 256>>>(h->d_q, h->d_bq[l], h->q_out);
        if (h->d_bk[l]) k_add_bias<<<(kv_dim + 255) / 256, 256>>>(kptr, h->d_bk[l], kv_dim);
        if (h->d_bv[l]) k_add_bias<<<(kv_dim + 255) / 256, 256>>>(vptr, h->d_bv[l], kv_dim);
        int n_kv = kv_dim / head_size;
        if (c->rope_neox) {
            k_rope_neox<<<(n_heads + 63) / 64, 64>>>(h->d_q, h->d_cos, h->d_sin, h->d_q, 1, n_heads, head_size);
            k_rope_neox<<<(n_kv + 63) / 64, 64>>>(kptr, h->d_cos, h->d_sin, kptr, 1, n_kv, head_size);
        } else {
            k_rope<<<(n_heads + 63) / 64, 64>>>(h->d_q, h->d_cos, h->d_sin, h->d_q, 1, n_heads, head_size);
            k_rope<<<(n_kv + 63) / 64, 64>>>(kptr, h->d_cos, h->d_sin, kptr, 1, n_kv, head_size);
        }
        k_attention<<<(n_heads + 63) / 64, 64>>>(h->d_q, h->d_keyc[l], h->d_valc[l], h->d_att, n_heads, head_size, kv_dim, c->kv_mul, pos);
        lin(h->d_att, h->d_wo[l], h->q6k_wo[l], h->d_x, dim, dim, 1, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);                 // wo + residual
        k_rms<<<1, 64>>>(h->d_x, h->d_rms_ffn[l], h->d_xb, dim, 1, c->rms_eps);
        lin(h->d_xb, h->d_w1[l], h->q6k_w1[l], h->d_gate, hidden, dim, 0, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);
        lin(h->d_xb, h->d_w3[l], h->q6k_w3[l], h->d_up, hidden, dim, 0, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);
        k_silu_mul<<<(hidden + 255) / 256, 256>>>(h->d_gate, h->d_up, h->d_inner, hidden);
        lin(h->d_inner, h->d_w2[l], h->q6k_w2[l], h->d_x, dim, hidden, 1, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);            // w2 + residual
        if (dbg && l <= 1) { char t[32]; snprintf(t, 32, "after layer %d", l); l1(t, h->d_x, dim); }
    }
    k_rms<<<1, 64>>>(h->d_x, h->d_rms_final, h->d_xb, dim, 1, c->rms_eps);
    if (dbg) l1("final rms xb", h->d_xb, dim);
    lin(h->d_xb, h->d_wcls, h->wcls_q6k, h->d_logits, c->vocab, dim, 0, h->d_q8_dx, h->d_q8_qs, h->regime, h->d_q8_bsums);
    if (dbg) l1("logits", h->d_logits, c->vocab);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return (int)e;
    cudaMemcpy(out_logits, h->d_logits, (size_t)c->vocab * 4, cudaMemcpyDeviceToHost);
    return 0;
}

// Isolation hook: run ONE linear of the given kind on the GPU (out[M] = W[M,K] . x[K]).
// Lets a host test compare a single quant kernel against the CPU dequant+canonical_dot.
extern "C" int vt_cuda_test_linear(int kind, int K, int M, const float* x, const unsigned char* w, size_t wlen, float* out, int iters) {
    float* dx = nullptr; unsigned char* dw = nullptr; float* dout = nullptr; int err = 0;
    dx = dev_floats(x, (size_t)K, &err);
    dw = dev_bytes(w, wlen, &err);
    dout = dev_zeros((size_t)M, &err);
    float* q8dx = dev_zeros(K / 256 > 0 ? K / 256 : 1, &err); signed char* q8qs = nullptr;
    if (cudaMalloc(&q8qs, (size_t)((K / 256 > 0 ? K / 256 : 1)) * 256) != cudaSuccess) err = 1;
    if (err) return -1;
    for (int it = 0; it < (iters < 1 ? 1 : iters); it++)
        lin(dx, dw, kind, dout, M, K, 0, q8dx, q8qs, 2, nullptr);   // regime-2 isolation test
    if (cudaDeviceSynchronize() != cudaSuccess) return -2;
    cudaMemcpy(out, dout, (size_t)M * 4, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dout); cudaFree(q8dx); cudaFree(q8qs);
    return 0;
}

// TIER-2 isolation hook: integer Q4_K linear. Quantizes x once, runs the dot `iters`
// times (for timing), returns out[M]. mode: 0 = plain integer loop, 1 = __dp4a.
extern "C" int vt_cuda_test_linear_int(int K, int M, const float* x, const unsigned char* w, size_t wlen,
                                       float* out, int mode, int iters) {
    int nsuper = K / 256; int err = 0;
    float* dx_in = dev_floats(x, (size_t)K, &err);
    unsigned char* dw = dev_bytes(w, wlen, &err);
    float* ddx = dev_zeros(nsuper, &err);
    float* dout = dev_zeros((size_t)M, &err);
    signed char* dqs = nullptr; int* dbs = nullptr;
    if (cudaMalloc(&dqs, (size_t)nsuper * 256) != cudaSuccess) err = 1;
    if (cudaMalloc(&dbs, (size_t)nsuper * 8 * 4) != cudaSuccess) err = 1;
    if (err) return -1;
    k_q4k_q8_quantize<<<(nsuper + 63) / 64, 64>>>(dx_in, ddx, dqs, dbs, nsuper);
    for (int it = 0; it < (iters < 1 ? 1 : iters); it++) {
        if (mode >= 2) k_q4k_q8_dot_warp<<<M, 32>>>(ddx, dqs, dbs, dw, dout, M, nsuper, mode & 1);  // warp-per-row (coalesced)
        else           k_q4k_q8_dot<<<(M + 63) / 64, 64>>>(ddx, dqs, dbs, dw, dout, M, nsuper, mode); // 1 thread/row
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return -2;
    cudaMemcpy(out, dout, (size_t)M * 4, cudaMemcpyDeviceToHost);
    cudaFree(dx_in); cudaFree(dw); cudaFree(ddx); cudaFree(dout); cudaFree(dqs); cudaFree(dbs);
    return 0;
}

// Batched prefill Q4_K linear test: out[N][M] (token-major) = W[M,K] . x[N,K], regime-3.
extern "C" int vt_cuda_test_prefill_q4k(int K, int M, int N, const float* x, const unsigned char* w, size_t wlen, float* out, int iters) {
    int ns = K / 256; int err = 0;
    float* dX = dev_floats(x, (size_t)N * K, &err);
    unsigned char* dW = dev_bytes(w, wlen, &err);
    float* ddx = dev_zeros((size_t)N * ns, &err);
    float* dout = dev_zeros((size_t)N * M, &err);
    signed char* dqs = nullptr; int* dbs = nullptr;
    if (cudaMalloc(&dqs, (size_t)N * ns * 256) != cudaSuccess) err = 1;
    if (cudaMalloc(&dbs, (size_t)N * ns * 8 * 4) != cudaSuccess) err = 1;
    if (err) return -1;
    k_q8_quant_batch<<<((size_t)N * ns + 127) / 128, 128>>>(dX, ddx, dqs, dbs, N, K, ns);
    dim3 grid((M + PBM - 1) / PBM, (N + PBN - 1) / PBN), blk(256);
    for (int it = 0; it < (iters < 1 ? 1 : iters); it++)
        k_q4k_rs_gemm<<<grid, blk>>>(dW, ddx, dqs, dbs, dout, M, N, ns, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return -2;
    cudaMemcpy(out, dout, (size_t)N * M * 4, cudaMemcpyDeviceToHost);
    cudaFree(dX); cudaFree(dW); cudaFree(ddx); cudaFree(dout); cudaFree(dqs); cudaFree(dbs);
    return 0;
}

extern "C" const char* vt_cuda_device_name() {
    static char name[256];
    cudaDeviceProp p; if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "unknown";
    snprintf(name, sizeof(name), "%s (CC %d.%d)", p.name, p.major, p.minor);
    return name;
}
