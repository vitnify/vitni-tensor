#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Q8_0 and Q5_0 linear kernels — dequantize each weight row on the fly and run
// the SAME canonical dot (8 lanes by i%8, fixed_tree per 8192-element chunk,
// fixed_tree over chunks) as q4k_linear / f32_matvec. Bit-identical to the CPU
// engine's Q8_0 / Q5_0 path (per-row dequant + canonical_dot) and to the CUDA
// k_q8_0 / k_q5_0. Regime-independent (these dtypes are unchanged across regimes).
// Needed for models that mix Q8_0 / Q5_0 with the k-quants (e.g. Qwen2.5).
// ===================================================================

constant uint CANON_LANES = 8u;
constant uint CANON_CHUNK = 8192u;

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

// One Q8_0 element: 34-byte block (f16 scale + 32 int8), value = scale * int8.
static inline float q8_0_val(device const uchar* wrow, uint idx) {
    uint blk = idx >> 5, within = idx & 31u;
    device const uchar* bp = wrow + blk * 34u;
    float d = f16_to_f32((ushort)((uint)bp[0] | ((uint)bp[1] << 8)));
    return d * (float)((char)bp[2u + within]);
}

// One Q5_0 element: 22-byte block (f16 scale + 4-byte high-bit field + 16 packed
// nibbles), value = scale * ((5-bit) - 16).
static inline float q5_0_val(device const uchar* wrow, uint idx) {
    uint blk = idx >> 5, within = idx & 31u;
    device const uchar* bp = wrow + blk * 22u;
    float scale = f16_to_f32((ushort)((uint)bp[0] | ((uint)bp[1] << 8)));
    uint qh = (uint)bp[2] | ((uint)bp[3] << 8) | ((uint)bp[4] << 16) | ((uint)bp[5] << 24);
    device const uchar* qs = bp + 6;
    int v;
    if (within < 16u) { uint j = within;       uint xh = ((qh >> j) << 4) & 0x10u;        v = (int)((qs[j] & 0x0Fu) | xh) - 16; }
    else              { uint j = within - 16u; uint xh = (qh >> (j + 12u)) & 0x10u;        v = (int)((qs[j] >> 4) | xh) - 16; }
    return scale * (float)v;
}

static inline float q8_0_dot(device const float* x, device const uchar* wrow, uint K) {
    float chunk_sums[64];
    uint nch = 0u, done = 0u;
    while (done < K) {
        uint e = min(done + CANON_CHUNK, K);
        uint n = e - done;
        float lanes[8];
        for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
        uint full = n - (n % CANON_LANES);
        uint i = 0u;
        for (; i < full; i += CANON_LANES) {
            for (uint j = 0u; j < CANON_LANES; j++) { float p = x[done + i + j] * q8_0_val(wrow, done + i + j); lanes[j] += p; }
        }
        for (; i < n; i++) { float p = x[done + i] * q8_0_val(wrow, done + i); lanes[i % CANON_LANES] += p; }
        chunk_sums[nch++] = fixed_tree(lanes, 8u);
        done = e;
    }
    return fixed_tree(chunk_sums, nch);
}

static inline float q5_0_dot(device const float* x, device const uchar* wrow, uint K) {
    float chunk_sums[64];
    uint nch = 0u, done = 0u;
    while (done < K) {
        uint e = min(done + CANON_CHUNK, K);
        uint n = e - done;
        float lanes[8];
        for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
        uint full = n - (n % CANON_LANES);
        uint i = 0u;
        for (; i < full; i += CANON_LANES) {
            for (uint j = 0u; j < CANON_LANES; j++) { float p = x[done + i + j] * q5_0_val(wrow, done + i + j); lanes[j] += p; }
        }
        for (; i < n; i++) { float p = x[done + i] * q5_0_val(wrow, done + i); lanes[i % CANON_LANES] += p; }
        chunk_sums[nch++] = fixed_tree(lanes, 8u);
        done = e;
    }
    return fixed_tree(chunk_sums, nch);
}

kernel void q8_0_linear(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],  // {K, M, nsuper(ignored)}
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1];
    if (gid >= M) return;
    out[gid] = q8_0_dot(x, w + (ulong)gid * (K >> 5) * 34u, K);
}
kernel void q8_0_linear_acc(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1];
    if (gid >= M) return;
    out[gid] = out[gid] + q8_0_dot(x, w + (ulong)gid * (K >> 5) * 34u, K);
}

kernel void q5_0_linear(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1];
    if (gid >= M) return;
    out[gid] = q5_0_dot(x, w + (ulong)gid * (K >> 5) * 22u, K);
}
kernel void q5_0_linear_acc(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1];
    if (gid >= M) return;
    out[gid] = out[gid] + q5_0_dot(x, w + (ulong)gid * (K >> 5) * 22u, K);
}
