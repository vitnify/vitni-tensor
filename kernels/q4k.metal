#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Q4_K fused dequant + canonical dot — GGUF inference hot path, on-GPU,
// bit-identical to vitni_tensor::ops::quant::canonical_dot_q4k_fused.
//
// One thread per output row: out[i] = canonical_dot_q4k_fused(x, W_row_i, K).
// Same numerical discipline as matmul: fixed reduction order (8 lanes by i%8,
// fixed pairwise tree, 8192-element chunks), separate rounded mul+add. The
// dequant `d*q - m` is an a*b-c pattern, which Metal WILL contract into an fnms
// even with fast-math off, so each product is forced to f32 (volatile) first.
// ===================================================================

constant uint CANON_LANES = 8u;
constant uint CANON_CHUNK = 8192u;
constant uint Q4K_BYTES   = 144u;
constant uint Q4K_NUMEL   = 256u;

// Exact port of vitni-tensor f16_to_f32 (integer bit-manip, so f16 subnormals
// and the whole range convert identically — not relying on Metal's half type,
// which may flush subnormals).
static inline float f16_to_f32(ushort h) {
    uint sign = (uint)((h >> 15) & 0x1u);
    uint expo = (uint)((h >> 10) & 0x1Fu);
    uint mant = (uint)(h & 0x3FFu);
    uint bits;
    if (expo == 0u) {
        if (mant == 0u) {
            bits = sign << 31;
        } else {
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

// (scale, min) 6-bit unpack, exact port of get_scale_min_k4.
static inline uint2 get_scale_min_k4(uint j, device const uchar* scales) {
    if (j < 4u) {
        return uint2((uint)(scales[j] & 0x3Fu), (uint)(scales[j + 4u] & 0x3Fu));
    }
    uint d = (uint)((scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6) << 4));
    uint m = (uint)((scales[j + 4u] >> 4)    | ((scales[j]      >> 6) << 4));
    return uint2(d, m);
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

// Plain f32 canonical dot: out[gid] = canonical_dot(x, W_row). Used when the
// Q4_K weights are PRE-DEQUANTIZED to f32 at load, so the per-token dequant is
// gone. Bit-identical to q4k_linear (which is dequant-then-canonical-dot).
static inline float canon_dot_f32(device const float* x, device const float* w, uint K) {
    float chunk_sums[64];
    uint nch = 0u;
    uint done = 0u;
    while (done < K) {
        uint e = min(done + CANON_CHUNK, K);
        uint n = e - done;
        float lanes[8];
        for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
        uint full = n - (n % CANON_LANES);
        uint i = 0u;
        for (; i < full; i += CANON_LANES) {
            for (uint j = 0u; j < CANON_LANES; j++) { float p = x[done + i + j] * w[done + i + j]; lanes[j] += p; }
        }
        for (; i < n; i++) { uint jj = i % CANON_LANES; float p = x[done + i] * w[done + i]; lanes[jj] += p; }
        chunk_sums[nch++] = fixed_tree(lanes, 8u);
        done = e;
    }
    return fixed_tree(chunk_sums, nch);
}
kernel void f32_matvec(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],  // {K, M}
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1];
    if (gid >= M) return;
    out[gid] = canon_dot_f32(x, w + (ulong)gid * K, K);
}
kernel void f32_matvec_acc(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1];
    if (gid >= M) return;
    out[gid] = out[gid] + canon_dot_f32(x, w + (ulong)gid * K, K);
}

// Accumulating variant: out[gid] += dot, fusing the residual add into the
// linear (removes a separate add_inplace dispatch). Bit-identical: `a + dot`
// with a = the residual value already in out[gid].
kernel void q4k_linear_acc(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q4K_BYTES;
    float lanes[8];
    for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
    float chunk_sums[64];
    uint nchunks_used = 0u;
    uint supers_per_chunk = CANON_CHUNK / Q4K_NUMEL;
    float buf[256];
    for (uint b = 0u; b < nsuper; b++) {
        uint base = b * Q4K_NUMEL;
        if (base >= K) break;
        device const uchar* blk = wrow + b * Q4K_BYTES;
        float d    = f16_to_f32((ushort)((uint)blk[0] | ((uint)blk[1] << 8)));
        float dmin = f16_to_f32((ushort)((uint)blk[2] | ((uint)blk[3] << 8)));
        device const uchar* scales = blk + 4;
        device const uchar* qs = blk + 16;
        uint is = 0u, q_off = 0u, y = 0u;
        for (uint sub = 0u; sub < 4u; sub++) {
            uint2 sm1 = get_scale_min_k4(is, scales);
            uint2 sm2 = get_scale_min_k4(is + 1u, scales);
            float d1 = d * (float)sm1.x; float m1f = dmin * (float)sm1.y;
            float d2 = d * (float)sm2.x; float m2f = dmin * (float)sm2.y;
            for (uint t = 0u; t < 32u; t++) {
                uchar q = qs[q_off + t];
                volatile float plo = d1 * (float)(q & 0x0Fu); buf[y + t]       = plo - m1f;
                volatile float phi = d2 * (float)(q >> 4);    buf[y + 32u + t] = phi - m2f;
            }
            y += 64u; q_off += 32u; is += 2u;
        }
        uint avail = min(Q4K_NUMEL, K - base);
        uint full = avail - (avail % CANON_LANES);
        for (uint i = 0u; i < full; i += CANON_LANES) {
            for (uint j = 0u; j < CANON_LANES; j++) { float p = x[base + i + j] * buf[i + j]; lanes[j] += p; }
        }
        for (uint t = full; t < avail; t++) { float p = x[base + t] * buf[t]; lanes[t % CANON_LANES] += p; }
        bool is_last = (base + avail >= K);
        if (((b + 1u) % supers_per_chunk == 0u) || is_last) {
            float l[8];
            for (uint j = 0u; j < 8u; j++) l[j] = lanes[j];
            chunk_sums[nchunks_used++] = fixed_tree(l, 8u);
            for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
        }
    }
    out[gid] = out[gid] + fixed_tree(chunk_sums, nchunks_used);
}

// Dequantize one Q4_K row to f32 (for the embedding lookup). One thread per
// super-block. Same guarded dequant as q4k_linear; no dot.
kernel void q4k_dequant(
    device const uchar* w    [[buffer(0)]],  // one row: nsuper*144 bytes
    device       float* out  [[buffer(1)]],  // nsuper*256 f32
    constant     uint*  dims [[buffer(2)]],  // {nsuper}
    uint b [[thread_position_in_grid]])
{
    uint nsuper = dims[0];
    if (b >= nsuper) return;
    device const uchar* blk = w + b * Q4K_BYTES;
    float d    = f16_to_f32((ushort)((uint)blk[0] | ((uint)blk[1] << 8)));
    float dmin = f16_to_f32((ushort)((uint)blk[2] | ((uint)blk[3] << 8)));
    device const uchar* scales = blk + 4;
    device const uchar* qs = blk + 16;
    device float* obuf = out + b * Q4K_NUMEL;
    uint is = 0u, q_off = 0u, y = 0u;
    for (uint sub = 0u; sub < 4u; sub++) {
        uint2 sm1 = get_scale_min_k4(is, scales);
        uint2 sm2 = get_scale_min_k4(is + 1u, scales);
        float d1 = d * (float)sm1.x; float m1f = dmin * (float)sm1.y;
        float d2 = d * (float)sm2.x; float m2f = dmin * (float)sm2.y;
        for (uint t = 0u; t < 32u; t++) {
            uchar q = qs[q_off + t];
            volatile float plo = d1 * (float)(q & 0x0Fu); obuf[y + t]       = plo - m1f;
            volatile float phi = d2 * (float)(q >> 4);    obuf[y + 32u + t] = phi - m2f;
        }
        y += 64u; q_off += 32u; is += 2u;
    }
}

kernel void q4k_linear(
    device const float* x    [[buffer(0)]],
    device const uchar* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],  // {K, M, nsuper}
    uint gid [[thread_position_in_grid]])
{
    uint K = dims[0], M = dims[1], nsuper = dims[2];
    if (gid >= M) return;
    device const uchar* wrow = w + (ulong)gid * nsuper * Q4K_BYTES;

    float lanes[8];
    for (uint j = 0u; j < 8u; j++) lanes[j] = 0.0f;
    float chunk_sums[64];
    uint nchunks_used = 0u;
    uint supers_per_chunk = CANON_CHUNK / Q4K_NUMEL; // 32
    float buf[256];

    for (uint b = 0u; b < nsuper; b++) {
        uint base = b * Q4K_NUMEL;
        if (base >= K) break;
        device const uchar* blk = wrow + b * Q4K_BYTES;
        float d    = f16_to_f32((ushort)((uint)blk[0] | ((uint)blk[1] << 8)));
        float dmin = f16_to_f32((ushort)((uint)blk[2] | ((uint)blk[3] << 8)));
        device const uchar* scales = blk + 4;
        device const uchar* qs = blk + 16;
        uint is = 0u, q_off = 0u, y = 0u;
        for (uint sub = 0u; sub < 4u; sub++) {
            uint2 sm1 = get_scale_min_k4(is, scales);
            uint2 sm2 = get_scale_min_k4(is + 1u, scales);
            float d1 = d * (float)sm1.x; float m1f = dmin * (float)sm1.y;
            float d2 = d * (float)sm2.x; float m2f = dmin * (float)sm2.y;
            for (uint t = 0u; t < 32u; t++) {
                uchar q = qs[q_off + t];
                volatile float plo = d1 * (float)(q & 0x0Fu); buf[y + t]       = plo - m1f;
                volatile float phi = d2 * (float)(q >> 4);    buf[y + 32u + t] = phi - m2f;
            }
            y += 64u; q_off += 32u; is += 2u;
        }
        uint avail = min(Q4K_NUMEL, K - base);
        uint full = avail - (avail % CANON_LANES);
        for (uint i = 0u; i < full; i += CANON_LANES) {
            for (uint j = 0u; j < CANON_LANES; j++) {
                float p = x[base + i + j] * buf[i + j]; // matmul-safe named product
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
