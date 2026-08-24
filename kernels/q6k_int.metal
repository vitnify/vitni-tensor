#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Q6_K INTEGER linear — bit-identical to vitni_tensor::ops::quant::
// linear_q6_k_integer, which is what the quantized forward actually calls for
// Q6_K weights (down_proj + lm_head). This is NOT the f32-dequant path: x is
// quantized to int8 per super-block, dotted with the Q6_K weights in integer
// arithmetic, then the per-super-block parts are combined with fixed_tree.
// Only div_sw (the cross-vendor software divide) touches floating point in the
// quantization; the dot is exact integer math.
// ===================================================================

constant uint Q6K_BYTES = 210u;
constant uint Q6K_NUMEL = 256u;

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
            uint m = mant; int e = 1;
            while ((m & 0x400u) == 0u) { m <<= 1; e -= 1; }
            m &= 0x3FFu;
            bits = (sign << 31) | ((uint)(127 - 15 + e) << 23) | (m << 13);
        }
    } else if (expo == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((expo + 127u - 15u) << 23) | (mant << 13);
    }
    return as_type<float>(bits);
}

static inline float fixed_tree(thread float* part, uint len) {
    if (len == 0u) return 0.0f;
    while (len > 1u) {
        uint hlen = (len + 1u) / 2u;
        for (uint t = 0u; t < hlen; t++) {
            uint u = 2u * t;
            part[t] = (u + 1u < len) ? (part[u] + part[u + 1u]) : part[u];
        }
        len = hlen;
    }
    return part[0];
}

// --- quantize-once path: the activation x is the same for every output row, so
// quantize it ONCE (q6k_quantize) and let the dot (q6k_integer_dot) reuse it,
// instead of re-quantizing x per row. Bit-identical to q6k_integer_linear.
kernel void q6k_quantize(
    device const float* x      [[buffer(0)]],  // [nsuper*256]
    device       float* out_dx [[buffer(1)]],  // [nsuper] per-block scales
    device       char*  out_qs [[buffer(2)]],  // [nsuper*256] int8
    constant     uint&  nsuper [[buffer(3)]],
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
    for (uint i = 0u; i < 256u; i++) {
        float sc = xb[i] * inv;
        float bias = (sc >= 0.0f) ? 0.5f : -0.5f;
        int q = (int)(sc + bias);
        q = clamp(q, -127, 127);
        qo[i] = (char)q;
    }
}

kernel void q6k_integer_dot(
    device const float* xdx  [[buffer(0)]],  // [nsuper]  (from q6k_quantize)
    device const char*  xqs  [[buffer(1)]],  // [nsuper*256]
    device const uchar* w    [[buffer(2)]],
    device       float* out  [[buffer(3)]],
    constant     uint*  dims [[buffer(4)]],  // {K, M, nsuper}
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q6K_BYTES;
    float parts[64];
    for (uint s = 0u; s < nsuper; s++) {
        device const char* qs = xqs + s * 256u;
        float dx = xdx[s];
        device const uchar* blk = wrow + s * Q6K_BYTES;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128;
        device const uchar* sc = blk + 192;
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
        parts[s] = d_w * dx * (float)acc;
    }
    out[gid] = fixed_tree(parts, nsuper);
}

// Accumulating variant of q6k_integer_dot: out[gid] += dot (fuses residual).
kernel void q6k_integer_dot_acc(
    device const float* xdx  [[buffer(0)]],
    device const char*  xqs  [[buffer(1)]],
    device const uchar* w    [[buffer(2)]],
    device       float* out  [[buffer(3)]],
    constant     uint*  dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q6K_BYTES;
    float parts[64];
    for (uint s = 0u; s < nsuper; s++) {
        device const char* qs = xqs + s * 256u;
        float dx = xdx[s];
        device const uchar* blk = wrow + s * Q6K_BYTES;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128;
        device const uchar* sc = blk + 192;
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
        parts[s] = d_w * dx * (float)acc;
    }
    out[gid] = out[gid] + fixed_tree(parts, nsuper);
}

kernel void q6k_integer_linear(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],  // {K, M, nsuper}
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q6K_BYTES;
    float parts[64];

    for (uint s = 0u; s < nsuper; s++) {
        device const float* xb = x + s * Q6K_NUMEL;
        // quantize_block_q8
        float amax = 0.0f;
        for (uint i = 0u; i < 256u; i++) { float a = fabs(xb[i]); if (a > amax) amax = a; }
        float dx = div_sw(amax, 127.0f);
        float inv = (dx != 0.0f) ? div_sw(1.0f, dx) : 0.0f;
        char qs[256];
        for (uint i = 0u; i < 256u; i++) {
            float scaled = xb[i] * inv;
            float bias = (scaled >= 0.0f) ? 0.5f : -0.5f;
            int q = (int)(scaled + bias);
            q = clamp(q, -127, 127);
            qs[i] = (char)q;
        }
        // dot_q6k_q8_block (full-block fast path; n == 256)
        device const uchar* blk = wrow + s * Q6K_BYTES;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128;
        device const uchar* sc = blk + 192;
        float d_w = f16_to_f32((ushort)((uint)blk[208] | ((uint)blk[209] << 8)));
        int sums[16];
        for (uint j = 0u; j < 16u; j++) sums[j] = 0;
        for (uint hb = 0u; hb < 2u; hb++) {
            uint ql_p = hb * 64u, qh_p = hb * 32u, sc_p = hb * 8u, y = hb * 128u;
            for (uint pass = 0u; pass < 2u; pass++) {
                uint is = pass, l0 = pass * 16u;
                int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
                for (uint l = l0; l < l0 + 16u; l++) {
                    uint h = (uint)qh[qh_p + l];
                    uint b0 = (uint)ql[ql_p + l];
                    uint b1 = (uint)ql[ql_p + l + 32u];
                    a1 += ((int)((b0 & 0x0Fu) | ((h & 0x03u) << 4)) - 32) * (int)qs[y + l];
                    a2 += ((int)((b1 & 0x0Fu) | (((h >> 2) & 0x03u) << 4)) - 32) * (int)qs[y + l + 32u];
                    a3 += ((int)((b0 >> 4) | (((h >> 4) & 0x03u) << 4)) - 32) * (int)qs[y + l + 64u];
                    a4 += ((int)((b1 >> 4) | (((h >> 6) & 0x03u) << 4)) - 32) * (int)qs[y + l + 96u];
                }
                sums[sc_p + is] += a1;
                sums[sc_p + is + 2u] += a2;
                sums[sc_p + is + 4u] += a3;
                sums[sc_p + is + 6u] += a4;
            }
        }
        int acc = 0;
        for (uint j = 0u; j < 16u; j++) { acc += (int)((char)sc[j]) * sums[j]; }
        parts[s] = d_w * dx * (float)acc;
    }
    out[gid] = fixed_tree(parts, nsuper);
}
