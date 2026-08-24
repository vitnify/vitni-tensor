#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Q6_K fused dequant + canonical dot — bit-identical to
// vitni_tensor::ops::quant::canonical_dot_q6k_fused. One thread per output row.
// Dequant is d*s*q (pure multiplies, no add/sub), so no contraction guard is
// needed; the reduction is the same canonical discipline as matmul/Q4_K.
// ===================================================================

constant uint CANON_LANES = 8u;
constant uint CANON_CHUNK = 8192u;
constant uint Q6K_BYTES   = 210u;
constant uint Q6K_NUMEL   = 256u;

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

kernel void q6k_linear(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],  // {K, M, nsuper}
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q6K_BYTES;

    float lanes[8];
    for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
    float chunk_sums[64];
    uint nchunks_used = 0u;
    uint supers_per_chunk = CANON_CHUNK / Q6K_NUMEL; // 32
    float buf[256];

    for (uint b = 0u; b < nsuper; b++) {
        uint base = b * Q6K_NUMEL;
        if (base >= K) break;
        device const uchar* blk = wrow + b * Q6K_BYTES;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128;
        device const uchar* sc = blk + 192;
        float d = f16_to_f32((ushort)((uint)blk[208] | ((uint)blk[209] << 8)));

        uint ql_p = 0u, qh_p = 0u, sc_p = 0u, y = 0u;
        for (uint hb = 0u; hb < 2u; hb++) { // 256 numel in two 128-blocks
            for (uint l = 0u; l < 32u; l++) {
                uint is = l / 16u;
                uint ql0 = (uint)ql[ql_p + l];
                uint ql1 = (uint)ql[ql_p + l + 32u];
                uint qhb = (uint)qh[qh_p + l];
                int q1 = (int)((ql0 & 0x0Fu) | ((qhb & 0x03u) << 4)) - 32;
                int q2 = (int)((ql1 & 0x0Fu) | (((qhb >> 2) & 0x03u) << 4)) - 32;
                int q3 = (int)((ql0 >> 4)    | (((qhb >> 4) & 0x03u) << 4)) - 32;
                int q4 = (int)((ql1 >> 4)    | (((qhb >> 6) & 0x03u) << 4)) - 32;
                float s1 = (float)((char)sc[sc_p + is]);
                float s2 = (float)((char)sc[sc_p + is + 2u]);
                float s3 = (float)((char)sc[sc_p + is + 4u]);
                float s4 = (float)((char)sc[sc_p + is + 6u]);
                buf[y + l]       = d * s1 * (float)q1;
                buf[y + l + 32u] = d * s2 * (float)q2;
                buf[y + l + 64u] = d * s3 * (float)q3;
                buf[y + l + 96u] = d * s4 * (float)q4;
            }
            y += 128u; ql_p += 64u; qh_p += 32u; sc_p += 8u;
        }

        uint avail = min(Q6K_NUMEL, K - base);
        uint full = avail - (avail % CANON_LANES);
        for (uint i = 0u; i < full; i += CANON_LANES) {
            for (uint j = 0u; j < CANON_LANES; j++) {
                float p = x[base + i + j] * buf[i + j];
                lanes[j] += p;
            }
        }
        for (uint t = full; t < avail; t++) {
            float p = x[base + t] * buf[t];
            lanes[t % CANON_LANES] += p;
        }
        bool is_last = (base + avail >= K);
        if (((b + 1u) % supers_per_chunk == 0u) || is_last) {
            float l[8];
            for (uint j = 0u; j < 8u; j++) l[j] = lanes[j];
            chunk_sums[nchunks_used++] = fixed_tree(l, 8u);
            for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
        }
    }
    out[gid] = fixed_tree(chunk_sums, nchunks_used);
}
