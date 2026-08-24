// Cross-vendor (NVIDIA) verification of the forward-pass kernels. Reads a golden
// file dumped on the Mac (inputs + CPU-reference outputs, incl. real TinyLlama
// Q4_K/Q6_K weights) and checks each CUDA kernel reproduces the CPU output
// bit-for-bit on the T4. If div_sw, expf, sqrt, q4k_linear and q6k_integer all
// match here, the full forward matches on NVIDIA by the same composition already
// demonstrated on Apple GPU.
//
// Build: nvcc --fmad=false -ftz=false -O2 -arch=sm_75 cuda_forward_conformance.cu -o cfc
//   --fmad=false  : no fma contraction (matches the CPU's separate mul/add)
//   -ftz=false    : keep denormals (correctly-rounded IEEE)
//
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// ------------------------ device helpers ------------------------
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

// ------------------------ kernels ------------------------
__global__ void k_div(const float* a, const float* b, float* out, unsigned n) { unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = div_sw(a[i], b[i]); }
__global__ void k_expf(const float* in, float* out, unsigned n) { unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = vt_expf(in[i]); }
__global__ void k_sqrt(const float* in, float* out, unsigned n) { unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = sqrtf(in[i]); }

__global__ void k_q4k(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    const unsigned char* wrow = w + (size_t)gid * nsuper * 144u;
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
            { unsigned j = is; if (j < 4u) { s1 = sc[j] & 0x3Fu; m1 = sc[j + 4u] & 0x3Fu; } else { s1 = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6) << 4); m1 = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4); } }
            { unsigned j = is + 1u; if (j < 4u) { s2 = sc[j] & 0x3Fu; m2 = sc[j + 4u] & 0x3Fu; } else { s2 = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6) << 4); m2 = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4); } }
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
    out[gid] = fixed_tree(chunk_sums, nch);
}

__global__ void k_q6k_int(const float* x, const unsigned char* w, float* out, unsigned K, unsigned M, unsigned nsuper) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= M) return;
    const unsigned char* wrow = w + (size_t)gid * nsuper * 210u;
    float parts[64];
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
    out[gid] = fixed_tree(parts, nsuper);
}

// rms_norm (regime-2): out = (x / sqrt(mean(x^2)+eps)) * w, canonical sumsq.
// One thread per row; feat < CANON_CHUNK so a single 8-lane chunk is exact.
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

// softmax over the last dim, one thread per row. Canonical-sum denominator.
__global__ void k_softmax(const float* x, float* out, unsigned last, unsigned rows) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; if (gid >= rows) return;
    const float* row = x + (size_t)gid * last; float* orow = out + (size_t)gid * last;
    float mx = -INFINITY; for (unsigned i = 0u; i < last; i++) { if (row[i] > mx) mx = row[i]; }
    for (unsigned i = 0u; i < last; i++) orow[i] = vt_expf(row[i] - mx);
    float lanes[8]; for (int j = 0; j < 8; j++) lanes[j] = 0.0f;
    unsigned full = last - (last % 8u), i = 0u;
    for (; i < full; i += 8u) for (unsigned j = 0u; j < 8u; j++) lanes[j] += orow[i + j];
    for (; i < last; i++) lanes[i % 8u] += orow[i];
    float l[8]; for (int j = 0; j < 8; j++) l[j] = lanes[j];
    float inv = div_sw(1.0f, fixed_tree(l, 8u));
    for (unsigned k = 0u; k < last; k++) orow[k] = orow[k] * inv;
}

// Multi-head attention (GQA + KV cache), one thread per query head. All three
// reductions (Q.K, softmax denom, A.V) are the canonical 8-lane+tree shape.
__global__ void k_attention(const float* q, const float* kc, const float* vc, float* xbout,
                            unsigned n_heads, unsigned head_size, unsigned kv_dim, unsigned kv_mul, unsigned pos) {
    unsigned h = blockIdx.x * blockDim.x + threadIdx.x; if (h >= n_heads) return;
    float scores[512];
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

// ------------------------ host ------------------------
static const unsigned char* g; static size_t gpos, glen;
static unsigned ru32() { unsigned v; memcpy(&v, g + gpos, 4); gpos += 4; return v; }
static const float* rf32(unsigned n) { const float* p = (const float*)(g + gpos); gpos += (size_t)n * 4; return p; }
static const unsigned char* rbytes(unsigned n) { const unsigned char* p = g + gpos; gpos += n; return p; }

template <class F> static int check(const char* name, const std::vector<float>& cpu, F run) {
    std::vector<float> gpu(cpu.size());
    run(gpu.data());
    size_t exact = 0; for (size_t i = 0; i < cpu.size(); i++) { unsigned a, b; memcpy(&a, &cpu[i], 4); memcpy(&b, &gpu[i], 4); if (a == b) exact++; }
    bool ok = exact == cpu.size();
    printf("  %-14s exact %zu/%zu  %s\n", name, exact, cpu.size(), ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "golden.bin";
    FILE* f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 2; }
    fseek(f, 0, SEEK_END); glen = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc(glen); fread(buf, 1, glen, f); fclose(f); g = buf; gpos = 0;
    if (ru32() != 0x564E5447u) { printf("bad magic\n"); return 2; }
    unsigned ncases = ru32();
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("== device: %s (CC %d.%d) ==\n== forward-kernel golden conformance (CUDA vs CPU) ==\n", prop.name, prop.major, prop.minor);
    int fails = 0;
    for (unsigned c = 0; c < ncases; c++) {
        unsigned op = ru32(); unsigned p0 = ru32(), p1 = ru32(), p2 = ru32(), p3 = ru32(); (void)p3;
        unsigned nin = ru32(); const float* in = rf32(nin);
        unsigned nw = ru32(); const unsigned char* w = rbytes(nw);
        unsigned no = ru32(); const float* outp = rf32(no);
        std::vector<float> cpu(outp, outp + no);
        if (op == 1) { // div: in = a[p0] ++ b[p0]
            unsigned n = p0; float *da, *db, *dr; cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dr, n * 4);
            cudaMemcpy(da, in, n * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, in + n, n * 4, cudaMemcpyHostToDevice);
            fails += check("div_sw", cpu, [&](float* o){ k_div<<<(n+255)/256,256>>>(da, db, dr, n); cudaDeviceSynchronize(); cudaMemcpy(o, dr, n*4, cudaMemcpyDeviceToHost); });
            cudaFree(da); cudaFree(db); cudaFree(dr);
        } else if (op == 2 || op == 3) {
            unsigned n = p0; float *di, *dobuf; cudaMalloc(&di, n * 4); cudaMalloc(&dobuf, n * 4); cudaMemcpy(di, in, n * 4, cudaMemcpyHostToDevice);
            const char* nm = op == 2 ? "expf" : "sqrt";
            fails += check(nm, cpu, [&](float* o){ if (op==2) k_expf<<<(n+255)/256,256>>>(di, dobuf, n); else k_sqrt<<<(n+255)/256,256>>>(di, dobuf, n); cudaDeviceSynchronize(); cudaMemcpy(o, dobuf, n*4, cudaMemcpyDeviceToHost); });
            cudaFree(di); cudaFree(dobuf);
        } else if (op == 4 || op == 5) {
            unsigned K = p0, M = p1, nsuper = p2;
            float* dx; unsigned char* dw; float* dy; cudaMalloc(&dx, nin * 4); cudaMalloc(&dw, nw); cudaMalloc(&dy, M * 4);
            cudaMemcpy(dx, in, nin * 4, cudaMemcpyHostToDevice); cudaMemcpy(dw, w, nw, cudaMemcpyHostToDevice);
            const char* nm = op == 4 ? "q4k_linear" : "q6k_integer";
            fails += check(nm, cpu, [&](float* o){ if (op==4) k_q4k<<<(M+63)/64,64>>>(dx, dw, dy, K, M, nsuper); else k_q6k_int<<<(M+63)/64,64>>>(dx, dw, dy, K, M, nsuper); cudaDeviceSynchronize(); cudaMemcpy(o, dy, M*4, cudaMemcpyDeviceToHost); });
            cudaFree(dx); cudaFree(dw); cudaFree(dy);
        } else if (op == 6) { // rms: in = x[feat*rows], w = weight[feat] as f32; p2 = eps bits
            unsigned feat = p0, rows = p1; float eps; memcpy(&eps, &p2, 4);
            float *dx, *dwt, *dout; cudaMalloc(&dx, nin*4); cudaMalloc(&dwt, nw); cudaMalloc(&dout, no*4);
            cudaMemcpy(dx, in, nin*4, cudaMemcpyHostToDevice); cudaMemcpy(dwt, w, nw, cudaMemcpyHostToDevice);
            fails += check("rms_norm", cpu, [&](float* o){ k_rms<<<(rows+63)/64,64>>>(dx, dwt, dout, feat, rows, eps); cudaDeviceSynchronize(); cudaMemcpy(o, dout, no*4, cudaMemcpyDeviceToHost); });
            cudaFree(dx); cudaFree(dwt); cudaFree(dout);
        } else if (op == 7) { // softmax: in = x[last*rows]
            unsigned last = p0, rows = p1;
            float *dx, *dout; cudaMalloc(&dx, nin*4); cudaMalloc(&dout, no*4); cudaMemcpy(dx, in, nin*4, cudaMemcpyHostToDevice);
            fails += check("softmax", cpu, [&](float* o){ k_softmax<<<(rows+63)/64,64>>>(dx, dout, last, rows); cudaDeviceSynchronize(); cudaMemcpy(o, dout, no*4, cudaMemcpyDeviceToHost); });
            cudaFree(dx); cudaFree(dout);
        } else if (op == 8) { // attention: in = q ++ kcache ++ vcache; p3 = (kv_mul<<16)|pos
            unsigned n_heads = p0, head_size = p1, kv_dim = p2, kv_mul = p3 >> 16, pos = p3 & 0xffffu;
            unsigned qn = n_heads * head_size, cn = (pos + 1u) * kv_dim;
            const float *q = in, *kc = in + qn, *vc = in + qn + cn;
            float *dq, *dk, *dv, *dout; cudaMalloc(&dq, qn*4); cudaMalloc(&dk, cn*4); cudaMalloc(&dv, cn*4); cudaMalloc(&dout, no*4);
            cudaMemcpy(dq, q, qn*4, cudaMemcpyHostToDevice); cudaMemcpy(dk, kc, cn*4, cudaMemcpyHostToDevice); cudaMemcpy(dv, vc, cn*4, cudaMemcpyHostToDevice);
            fails += check("attention", cpu, [&](float* o){ k_attention<<<(n_heads+63)/64,64>>>(dq, dk, dv, dout, n_heads, head_size, kv_dim, kv_mul, pos); cudaDeviceSynchronize(); cudaMemcpy(o, dout, no*4, cudaMemcpyDeviceToHost); });
            cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
        }
    }
    printf("VERDICT: %s\n", fails == 0 ? "PASS - all forward kernels bit-identical to CPU on NVIDIA" : "FAIL");
    return fails == 0 ? 0 : 1;
}
