#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Forward-pass reduction ops: rms_norm, softmax, silu — cross-vendor
// bit-identical to vitni_tensor's CPU ops. Built from the proven primitives
// (software division, expf) plus serial reductions in the CPU's exact order.
// Compile with fastMathEnabled=false; every fused a±b*c site is guarded.
// Helpers are copied from expf.metal (kept in sync by the conformance tests).
// ===================================================================

static inline float vt_scalbnf(float x, int n) {
    float f_exp_max     = as_type<float>((uint)254u << 23);
    float f_exp_min     = as_type<float>((uint)1u   << 23);
    float f_pow_subnorm = as_type<float>((uint)151u << 23);
    if (n > 127) {
        x = x * f_exp_max; n -= 127;
        if (n > 127) { x = x * f_exp_max; n -= 127; if (n > 127) n = 127; }
    } else if (n < -126) {
        float mul = f_exp_min * f_pow_subnorm; int add = 126 - 24;
        x = x * mul; n += add;
        if (n < -126) { x = x * mul; n += add; if (n < -126) n = -126; }
    }
    float scale = as_type<float>((uint)((127 + n)) << 23);
    return x * scale;
}

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

static inline float vt_expf(float x) {
    const float HALF0 = 0.5f, HALF1 = -0.5f;
    const float LN2_HI  = 6.9314575195e-01f;
    const float LN2_LO  = 1.4286067653e-06f;
    const float INV_LN2 = 1.4426950216e+00f;
    const float P1 = 1.6666625440e-1f;
    const float P2 = -2.7667332906e-3f;
    float x1p127 = as_type<float>((uint)0x7f000000u);
    uint hx = as_type<uint>(x);
    int sign = (int)(hx >> 31);
    bool signb = sign != 0;
    hx &= 0x7fffffffu;
    if (hx >= 0x42aeac50u) {
        if (hx > 0x7f800000u) return x;
        if ((hx >= 0x42b17218u) && !signb) { x = x * x1p127; return x; }
        if (signb) { if (hx >= 0x42cff1b5u) return 0.0f; }
    }
    int k; float hi, lo;
    if (hx > 0x3eb17218u) {
        if (hx > 0x3f851592u) {
            float h = signb ? HALF1 : HALF0;
            volatile float ix = INV_LN2 * x;
            k = (int)(ix + h);
        } else { k = 1 - sign - sign; }
        float kf = (float)k;
        volatile float khi = kf * LN2_HI;
        hi = x - khi;
        lo = kf * LN2_LO;
        x = hi - lo;
    } else if (hx > 0x39000000u) { k = 0; hi = x; lo = 0.0f; }
    else { return 1.0f + x; }
    float xx = x * x;
    volatile float m1 = xx * P2;
    float t = P1 + m1;
    volatile float m2 = xx * t;
    float c = x - m2;
    float xc = x * c;
    float y = 1.0f + ((div_sw(xc, 2.0f - c) - lo) + hi);
    return (k == 0) ? y : vt_scalbnf(y, k);
}

// Diagnostic: is Metal's sqrt correctly-rounded (matches libm::sqrtf / hardware)?
kernel void sqrt_kernel(
    device const float* in [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     uint&  n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = sqrt(in[gid]);
}

// rms_norm: out = (x / sqrt(mean(x^2)+eps)) * w, along the last dim.
kernel void rms_kernel(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],  // {feat, rows}
    constant     float& eps  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint feat = dims[0], rows = dims[1];
    if (gid >= rows) return;
    device const float* row = x + (ulong)gid * feat;
    float sumsq = 0.0f;
    for (uint i = 0u; i < feat; i++) {
        float p = row[i] * row[i]; // named product (matmul-safe += pattern)
        sumsq += p;
    }
    float mean = div_sw(sumsq, (float)feat);
    float scale = div_sw(1.0f, sqrt(mean + eps));
    device float* orow = out + (ulong)gid * feat;
    for (uint i = 0u; i < feat; i++) {
        float t = row[i] * scale;  // (v*scale)*w[i], two plain muls
        orow[i] = t * w[i];
    }
}

// softmax along the last dim: subtract max, exp, normalize.
kernel void softmax_kernel(
    device const float* x    [[buffer(0)]],
    device       float* out  [[buffer(1)]],
    constant     uint*  dims [[buffer(2)]],  // {last, rows}
    uint gid [[thread_position_in_grid]])
{
    uint last = dims[0], rows = dims[1];
    if (gid >= rows) return;
    device const float* row = x + (ulong)gid * last;
    device float* orow = out + (ulong)gid * last;
    float mx = -INFINITY;
    for (uint i = 0u; i < last; i++) { if (row[i] > mx) mx = row[i]; }
    float sum = 0.0f;
    for (uint i = 0u; i < last; i++) {
        float e = vt_expf(row[i] - mx);
        orow[i] = e;
        sum += e;
    }
    float inv = div_sw(1.0f, sum);
    for (uint i = 0u; i < last; i++) { orow[i] = orow[i] * inv; }
}

// RoPE apply: (a,b) -> (a*c - b*s, a*s + b*c), reading a CPU-precomputed
// cos/sin cache (the sinf/cosf that build it use f64 and stay on the CPU).
// One thread per (seq, head). Products are guarded so the a*c-b*s / a*s+b*c
// stay separately rounded, matching the CPU's non-fused form.
kernel void rope_apply(
    device const float* x    [[buffer(0)]],
    device const float* cosc [[buffer(1)]],
    device const float* sinc [[buffer(2)]],
    device       float* out  [[buffer(3)]],
    constant     uint*  dims [[buffer(4)]],  // {seq, n_heads, head_dim}
    uint gid [[thread_position_in_grid]])
{
    uint seq = dims[0], n_heads = dims[1], head_dim = dims[2];
    if (gid >= seq * n_heads) return;
    uint s = gid / n_heads;
    uint hd = head_dim / 2u;        // 'half' is a reserved MSL type name
    uint head_off = gid * head_dim;
    for (uint i = 0u; i < hd; i++) {
        float a = x[head_off + 2u * i];
        float b = x[head_off + 2u * i + 1u];
        float c = cosc[s * hd + i];
        float sn = sinc[s * hd + i];
        volatile float p1 = a * c; volatile float p2 = b * sn;
        out[head_off + 2u * i] = p1 - p2;
        volatile float p3 = a * sn; volatile float p4 = b * c;
        out[head_off + 2u * i + 1u] = p3 + p4;
    }
}

// Multi-head attention with KV cache, one thread per query head. Matches the
// CPU forward's PLAIN serial reductions (not the canonical lane/tree): serial
// Q.K dot, /sqrt(head_size), serial softmax, serial A.V accumulation over t.
// GQA: query head h reads kv head h/kv_mul. Caps context at 512 positions
// (enough for the conformance driver; raise if needed).
kernel void attention(
    device const float* q      [[buffer(0)]],  // [n_heads*head_size], this token, post-rope
    device const float* kcache [[buffer(1)]],  // [(pos+1)*kv_dim] for this layer
    device const float* vcache [[buffer(2)]],  // [(pos+1)*kv_dim] for this layer
    device       float* xbout  [[buffer(3)]],  // [n_heads*head_size]
    constant     uint*  d      [[buffer(4)]],  // {n_heads, head_size, kv_dim, kv_mul, pos}
    uint h [[thread_position_in_grid]])
{
    uint n_heads = d[0], head_size = d[1], kv_dim = d[2], kv_mul = d[3], pos = d[4];
    if (h >= n_heads) return;
    float scores[512];
    uint qoff = h * head_size;
    uint kvhead = h / kv_mul;
    float rsqrt_hs = sqrt((float)head_size);
    for (uint t = 0u; t <= pos; t++) {
        uint koff = t * kv_dim + kvhead * head_size;
        float sc = 0.0f;
        for (uint dd = 0u; dd < head_size; dd++) {
            float p = q[qoff + dd] * kcache[koff + dd]; // plain serial dot (named product)
            sc += p;
        }
        scores[t] = div_sw(sc, rsqrt_hs); // score / sqrt(head_size)
    }
    // serial softmax (== softmax_inplace)
    float mx = -INFINITY;
    for (uint t = 0u; t <= pos; t++) { if (scores[t] > mx) mx = scores[t]; }
    float sum = 0.0f;
    for (uint t = 0u; t <= pos; t++) { float e = vt_expf(scores[t] - mx); scores[t] = e; sum += e; }
    float inv = div_sw(1.0f, sum);
    for (uint t = 0u; t <= pos; t++) { scores[t] = scores[t] * inv; }
    // serial A.V (t outer, accumulate into xbout[d])
    for (uint dd = 0u; dd < head_size; dd++) xbout[qoff + dd] = 0.0f;
    for (uint t = 0u; t <= pos; t++) {
        uint voff = t * kv_dim + kvhead * head_size;
        float a = scores[t];
        for (uint dd = 0u; dd < head_size; dd++) {
            float p = a * vcache[voff + dd]; // named product
            xbout[qoff + dd] += p;
        }
    }
}

// SwiGLU inner: silu(gate) * up, fused so `up` never leaves the GPU.
kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint&  n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    out[gid] = div_sw(g, 1.0f + vt_expf(-g)) * up[gid];
}

// Residual add, in place: x[i] += y[i]. (Plain add — matches the CPU.)
kernel void add_inplace(
    device       float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant     uint&  n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = x[gid] + y[gid];
}

// silu(x) = x / (1 + e^-x)
kernel void silu_kernel(
    device const float* x   [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     uint&  n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    out[gid] = div_sw(v, 1.0f + vt_expf(-v));
}
