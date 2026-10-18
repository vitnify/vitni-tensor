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

// ============================ host driver (C ABI) ============================

typedef struct {
    int dim, hidden, n_layers, n_heads, head_size, kv_dim, kv_mul, vocab, max_seq;
    int rope_neox;               // 1 = NeoX split-half rotation (Qwen2); 0 = interleaved (Llama)
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
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) h->err = 1;
    return h->err;
}

// kind: 0 = Q4_K, 1 = Q6_K, 2 = Q8_0, 3 = Q5_0
static inline void lin(const float* in, const unsigned char* w, int kind, float* out, int M, int K, int acc) {
    int nsuper = K / 256;
    dim3 g((M + 63) / 64), b(64);
    if (kind == 1)      { if (acc) k_q6k_acc<<<g, b>>>(in, w, out, K, M, nsuper); else k_q6k<<<g, b>>>(in, w, out, K, M, nsuper); }
    else if (kind == 2) { if (acc) k_q8_0_acc<<<g, b>>>(in, w, out, K, M);        else k_q8_0<<<g, b>>>(in, w, out, K, M); }
    else if (kind == 3) { if (acc) k_q5_0_acc<<<g, b>>>(in, w, out, K, M);        else k_q5_0<<<g, b>>>(in, w, out, K, M); }
    else                { if (acc) k_q4k_acc<<<g, b>>>(in, w, out, K, M, nsuper); else k_q4k<<<g, b>>>(in, w, out, K, M, nsuper); }
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
        lin(h->d_xb, h->d_wq[l], h->q6k_wq[l], h->d_q, h->q_out, dim, 0);
        float* kptr = h->d_keyc[l] + (size_t)pos * kv_dim;
        float* vptr = h->d_valc[l] + (size_t)pos * kv_dim;
        lin(h->d_xb, h->d_wk[l], h->q6k_wk[l], kptr, kv_dim, dim, 0);
        lin(h->d_xb, h->d_wv[l], h->q6k_wv[l], vptr, kv_dim, dim, 0);
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
        lin(h->d_att, h->d_wo[l], h->q6k_wo[l], h->d_x, dim, dim, 1);                 // wo + residual
        k_rms<<<1, 64>>>(h->d_x, h->d_rms_ffn[l], h->d_xb, dim, 1, c->rms_eps);
        lin(h->d_xb, h->d_w1[l], h->q6k_w1[l], h->d_gate, hidden, dim, 0);
        lin(h->d_xb, h->d_w3[l], h->q6k_w3[l], h->d_up, hidden, dim, 0);
        k_silu_mul<<<(hidden + 255) / 256, 256>>>(h->d_gate, h->d_up, h->d_inner, hidden);
        lin(h->d_inner, h->d_w2[l], h->q6k_w2[l], h->d_x, dim, hidden, 1);            // w2 + residual
        if (dbg && l <= 1) { char t[32]; snprintf(t, 32, "after layer %d", l); l1(t, h->d_x, dim); }
    }
    k_rms<<<1, 64>>>(h->d_x, h->d_rms_final, h->d_xb, dim, 1, c->rms_eps);
    if (dbg) l1("final rms xb", h->d_xb, dim);
    lin(h->d_xb, h->d_wcls, h->wcls_q6k, h->d_logits, c->vocab, dim, 0);
    if (dbg) l1("logits", h->d_logits, c->vocab);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return (int)e;
    cudaMemcpy(out_logits, h->d_logits, (size_t)c->vocab * 4, cudaMemcpyDeviceToHost);
    return 0;
}

// Isolation hook: run ONE linear of the given kind on the GPU (out[M] = W[M,K] . x[K]).
// Lets a host test compare a single quant kernel against the CPU dequant+canonical_dot.
extern "C" int vt_cuda_test_linear(int kind, int K, int M, const float* x, const unsigned char* w, size_t wlen, float* out) {
    float* dx = nullptr; unsigned char* dw = nullptr; float* dout = nullptr; int err = 0;
    dx = dev_floats(x, (size_t)K, &err);
    dw = dev_bytes(w, wlen, &err);
    dout = dev_zeros((size_t)M, &err);
    if (err) return -1;
    lin(dx, dw, kind, dout, M, K, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return -2;
    cudaMemcpy(out, dout, (size_t)M * 4, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
    return 0;
}

extern "C" const char* vt_cuda_device_name() {
    static char name[256];
    cudaDeviceProp p; if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "unknown";
    snprintf(name, sizeof(name), "%s (CC %d.%d)", p.name, p.major, p.minor);
    return name;
}
