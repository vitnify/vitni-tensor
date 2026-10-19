#include <metal_stdlib>
using namespace metal;

// ===================================================================
// regime-3 (reserved): integer-activation k-quant linears with a fixed-order
// RUNNING-SUM combine across super-blocks. Bit-identical to the CPU engine's
// vitni_tensor::ops::quant::linear_q4_k_integer_rs / linear_q6_k_integer_rs, and
// to the CUDA backend's k_q4k_rs / k_q6k_rs. Same integer dot as regime-2's Q6_K
// path; the ONLY difference from regime-2 is the combine (running sum vs
// fixed_tree) and that Q4_K also runs in exact integers (regime-2 Q4_K is float).
// ===================================================================

constant uint Q4K_BYTES = 144u;
constant uint Q6K_BYTES = 210u;

static inline float div_sw(float a, float b) {
    uint sgn = (as_type<uint>(a) ^ as_type<uint>(b)) & 0x80000000u;
    float ba = as_type<float>(as_type<uint>(b) & 0x7fffffffu);
    float aa = as_type<float>(as_type<uint>(a) & 0x7fffffffu);
    uint j = 0x7EF127EAu - as_type<uint>(ba);
    float y = as_type<float>(j);
    float nb = -ba;
    float e;
    e = fma(nb, y, 1.0f); y = fma(y, e, y);
    e = fma(nb, y, 1.0f); y = fma(y, e, y);
    e = fma(nb, y, 1.0f); y = fma(y, e, y);
    float q = aa * y;
    float r = fma(nb, q, aa);
    q = fma(r, y, q);
    return as_type<float>(as_type<uint>(q) ^ sgn);
}

static inline float f16_to_f32(ushort h) {
    uint sign = (uint)((h >> 15) & 0x1u);
    uint expo = (uint)((h >> 10) & 0x1Fu);
    uint mant = (uint)(h & 0x3FFu);
    uint bits;
    if (expo == 0u) {
        if (mant == 0u) { bits = sign << 31; }
        else {
            uint m = mant; int ee = 1;
            while ((m & 0x400u) == 0u) { m <<= 1; ee -= 1; }
            m &= 0x3FFu;
            bits = (sign << 31) | ((uint)(127 - 15 + ee) << 23) | (m << 13);
        }
    } else if (expo == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((expo + 127u - 15u) << 23) | (mant << 13);
    }
    return as_type<float>(bits);
}

// 6-bit (scale,min) unpack for Q4_K — exact port of get_scale_min_k4.
static inline void get_scale_min_k4(uint j, device const uchar* sc, thread uint& s, thread uint& m) {
    if (j < 4u) { s = sc[j] & 0x3Fu; m = sc[j + 4u] & 0x3Fu; }
    else {
        s = (sc[j + 4u] & 0x0Fu) | ((uint)(sc[j - 4u] >> 6) << 4);
        m = (uint)(sc[j + 4u] >> 4) | ((uint)(sc[j] >> 6) << 4);
    }
}

// Quantize activation x to int8 per 256-elem super-block: scale dx, int8 qs, and
// per-32-group sums bsums (Q4_K's dmin term needs them; Q6_K ignores them).
// Matches quantize_block_q8 / the CUDA k_q4k_q8_quantize.
kernel void q8_quant(
    device const float* x       [[buffer(0)]],
    device       float* out_dx  [[buffer(1)]],
    device       char*  out_qs  [[buffer(2)]],
    device       int*   out_bs  [[buffer(3)]],
    constant     uint&  nsuper  [[buffer(4)]],
    uint s [[thread_position_in_grid]])
{
    if (s >= nsuper) return;
    device const float* xb = x + s * 256u;
    float amax = 0.0f;
    for (uint i = 0u; i < 256u; i++) { float a = fabs(xb[i]); if (a > amax) amax = a; }
    float dx = div_sw(amax, 127.0f);
    float inv = (dx != 0.0f) ? div_sw(1.0f, dx) : 0.0f;
    out_dx[s] = dx;
    device char* qo = out_qs + s * 256u;
    device int*  bs = out_bs + s * 8u;
    for (uint g = 0u; g < 8u; g++) bs[g] = 0;
    for (uint i = 0u; i < 256u; i++) {
        float sc = xb[i] * inv;
        float bias = (sc >= 0.0f) ? 0.5f : -0.5f;
        int q = (int)(sc + bias);
        q = clamp(q, -127, 127);
        qo[i] = (char)q;
        bs[i / 32u] += q;
    }
}

// One Q4_K super-block × a Q8 x-block, exact integer — matches q4k_q8_part.
static inline float q4k_q8_part(float d_x, device const char* xq, device const int* bsums, device const uchar* blk) {
    float d = f16_to_f32((ushort)((uint)blk[0] | ((uint)blk[1] << 8)));
    float dmin = f16_to_f32((ushort)((uint)blk[2] | ((uint)blk[3] << 8)));
    device const uchar* sc = blk + 4; device const uchar* qs = blk + 16;
    int acc = 0, acc_m = 0;
    for (uint hf = 0u; hf < 4u; hf++) {
        uint s1, m1, s2, m2;
        get_scale_min_k4(hf * 2u, sc, s1, m1);
        get_scale_min_k4(hf * 2u + 1u, sc, s2, m2);
        uint q_off = hf * 32u, base = hf * 64u;
        int s_lo = 0, s_hi = 0;
        for (uint l = 0u; l < 32u; l++) {
            s_lo += (int)(qs[q_off + l] & 0x0F) * (int)xq[base + l];
            s_hi += (int)(qs[q_off + l] >> 4)   * (int)xq[base + 32u + l];
        }
        acc   += (int)s1 * s_lo + (int)s2 * s_hi;
        acc_m += (int)m1 * bsums[hf * 2u] + (int)m2 * bsums[hf * 2u + 1u];
    }
    float a1 = d * d_x, a2 = dmin * d_x;      // separate rounding (fast-math off)
    return a1 * (float)acc - a2 * (float)acc_m;
}

// One Q6_K super-block × a Q8 x-block, exact integer — matches q6k_q8_part.
static inline float q6k_q8_part(float dx, device const char* qs, device const uchar* blk) {
    device const uchar* ql = blk; device const uchar* qh = blk + 128; device const uchar* sc = blk + 192;
    float d_w = f16_to_f32((ushort)((uint)blk[208] | ((uint)blk[209] << 8)));
    int sums[16];
    for (uint j = 0u; j < 16u; j++) sums[j] = 0;
    for (uint hb = 0u; hb < 2u; hb++) {
        uint ql_p = hb * 64u, qh_p = hb * 32u, sc_p = hb * 8u, y = hb * 128u;
        for (uint pass = 0u; pass < 2u; pass++) {
            uint is = pass, l0 = pass * 16u;
            int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
            for (uint l = l0; l < l0 + 16u; l++) {
                uint h = (uint)qh[qh_p + l], b0 = (uint)ql[ql_p + l], b1 = (uint)ql[ql_p + l + 32u];
                a1 += ((int)((b0 & 0x0Fu) | ((h & 0x03u) << 4)) - 32) * (int)qs[y + l];
                a2 += ((int)((b1 & 0x0Fu) | (((h >> 2) & 0x03u) << 4)) - 32) * (int)qs[y + l + 32u];
                a3 += ((int)((b0 >> 4) | (((h >> 4) & 0x03u) << 4)) - 32) * (int)qs[y + l + 64u];
                a4 += ((int)((b1 >> 4) | (((h >> 6) & 0x03u) << 4)) - 32) * (int)qs[y + l + 96u];
            }
            sums[sc_p + is] += a1; sums[sc_p + is + 2u] += a2; sums[sc_p + is + 4u] += a3; sums[sc_p + is + 6u] += a4;
        }
    }
    int acc = 0; for (uint j = 0u; j < 16u; j++) acc += (int)((char)sc[j]) * sums[j];
    return d_w * dx * (float)acc;
}

// Q4_K integer linear, running-sum combine. out[gid] = Σ_s part_s (left-to-right).
kernel void q4k_integer_rs(
    device const float* dx   [[buffer(0)]],
    device const char*  qs   [[buffer(1)]],
    device const int*   bs   [[buffer(2)]],
    device const uchar* w    [[buffer(3)]],
    device       float* out  [[buffer(4)]],
    constant     uint*  dims [[buffer(5)]],   // {K, M, nsuper}
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q4K_BYTES;
    float sum = 0.0f;
    for (uint s = 0u; s < nsuper; s++)
        sum = sum + q4k_q8_part(dx[s], qs + s * 256u, bs + s * 8u, wrow + s * Q4K_BYTES);
    out[gid] = sum;
}
kernel void q4k_integer_rs_acc(
    device const float* dx   [[buffer(0)]],
    device const char*  qs   [[buffer(1)]],
    device const int*   bs   [[buffer(2)]],
    device const uchar* w    [[buffer(3)]],
    device       float* out  [[buffer(4)]],
    constant     uint*  dims [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q4K_BYTES;
    float sum = 0.0f;
    for (uint s = 0u; s < nsuper; s++)
        sum = sum + q4k_q8_part(dx[s], qs + s * 256u, bs + s * 8u, wrow + s * Q4K_BYTES);
    out[gid] = out[gid] + sum;
}

// Q6_K integer linear, running-sum combine.
kernel void q6k_integer_rs(
    device const float* dx   [[buffer(0)]],
    device const char*  qs   [[buffer(1)]],
    device const uchar* w    [[buffer(2)]],
    device       float* out  [[buffer(3)]],
    constant     uint*  dims [[buffer(4)]],   // {K, M, nsuper}
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q6K_BYTES;
    float sum = 0.0f;
    for (uint s = 0u; s < nsuper; s++)
        sum = sum + q6k_q8_part(dx[s], qs + s * 256u, wrow + s * Q6K_BYTES);
    out[gid] = sum;
}
kernel void q6k_integer_rs_acc(
    device const float* dx   [[buffer(0)]],
    device const char*  qs   [[buffer(1)]],
    device const uchar* w    [[buffer(2)]],
    device       float* out  [[buffer(3)]],
    constant     uint*  dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q6K_BYTES;
    float sum = 0.0f;
    for (uint s = 0u; s < nsuper; s++)
        sum = sum + q6k_q8_part(dx[s], qs + s * 256u, wrow + s * Q6K_BYTES);
    out[gid] = out[gid] + sum;
}
