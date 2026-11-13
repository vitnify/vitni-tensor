#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Forward-pass reduction ops: rms_norm, softmax, attention — cross-vendor
// bit-identical to vitni_tensor's CPU ops. Built from the proven primitives
// (software division, expf) plus the CANONICAL reduction (regime-2): the
// same lane-pinned + fixed-tree shape ops::quant::canonical_dot uses, so a
// reduction is bit-identical regardless of thread count or vector width AND
// exposes lane parallelism (the serial accumulators these replaced pinned
// each row to one thread). Compile with fastMathEnabled=false; every fused
// a±b*c site is guarded. Helpers copied from expf.metal (conformance-synced).
// ===================================================================

// Fixed pairwise tree over CANON_LANES=8 lanes — bit-identical to
// ops::matmul::fixed_tree for n=8: ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)).
static inline float ftree8(thread float* lane) {
    float t0 = lane[0] + lane[1];
    float t1 = lane[2] + lane[3];
    float t2 = lane[4] + lane[5];
    float t3 = lane[6] + lane[7];
    float u0 = t0 + t1;
    float u1 = t2 + t3;
    return u0 + u1;
}

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
    // The reciprocal seed below only handles finite divisors. In particular,
    // SiLU's exp(-x) can overflow for finite negative x: CPU x / +inf is
    // signed zero. Preserve those bits instead of refining an invalid seed.
    if ((as_type<uint>(b) & 0x7fffffffu) == 0x7f800000u &&
        (as_type<uint>(a) & 0x7fffffffu) < 0x7f800000u) {
        return as_type<float>(sgn);
    }
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
    // sum(x*x) via the canonical lane+tree shape == canonical_dot(row, row).
    // feat (model dim) < CANON_CHUNK=8192, so a single chunk is bit-exact.
    float lane[8] = {0,0,0,0,0,0,0,0};
    uint full = feat - (feat % 8u);
    uint i = 0u;
    for (; i < full; i += 8u) {
        for (uint j = 0u; j < 8u; j++) {
            float p = row[i + j] * row[i + j]; // named product (no FMA contraction)
            lane[j] += p;
        }
    }
    for (; i < feat; i++) {
        uint j = i % 8u;
        float p = row[i] * row[i];
        lane[j] += p;
    }
    float sumsq = ftree8(lane);
    float mean = div_sw(sumsq, (float)feat);
    float scale = div_sw(1.0f, sqrt(mean + eps));
    device float* orow = out + (ulong)gid * feat;
    for (uint i = 0u; i < feat; i++) {
        float t = row[i] * scale;  // (v*scale)*w[i], two plain muls
        orow[i] = t * w[i];
    }
}

// Opt-in implementation of the SAME single-chunk RMS contract (feat <= 8192).
// A full 64-thread group owns one row. Only the eight canonical lanes reduce;
// every lane retains its element order. Thread 0 applies the original tree and
// scale arithmetic; all threads then write independent output elements.
kernel void rms_kernel_cooperative(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    constant     float& eps  [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint feat = dims[0], rows = dims[1], r = group.x;
    // Uniform across the entire group, so this cannot strand a barrier.
    if (r >= rows) return;
    device const float* row = x + (ulong)r * feat;
    threadgroup float partials[8];
    threadgroup float shared_scale;
    if (tid < 8u) {
        float sum = 0.0f;
        for (uint i = tid; i < feat; i += 8u) {
            volatile float product = row[i] * row[i];
            sum += product;
        }
        partials[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float lanes[8];
        for (uint j = 0u; j < 8u; j++) lanes[j] = partials[j];
        float sumsq = ftree8(lanes);
        float mean = div_sw(sumsq, (float)feat);
        shared_scale = div_sw(1.0f, sqrt(mean + eps));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_scale;
    device float* orow = out + (ulong)r * feat;
    for (uint i = tid; i < feat; i += 64u) {
        volatile float scaled = row[i] * scale;
        orow[i] = scaled * w[i];
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
    for (uint i = 0u; i < last; i++) { orow[i] = vt_expf(row[i] - mx); }
    // denominator via canonical lane+tree sum (== canonical_sum on CPU).
    float lane[8] = {0,0,0,0,0,0,0,0};
    uint full = last - (last % 8u);
    uint i = 0u;
    for (; i < full; i += 8u) { for (uint j = 0u; j < 8u; j++) lane[j] += orow[i + j]; }
    for (; i < last; i++) { lane[i % 8u] += orow[i]; }
    float sum = ftree8(lane);
    float inv = div_sw(1.0f, sum);
    for (uint k = 0u; k < last; k++) { orow[k] = orow[k] * inv; }
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

// NeoX RoPE (Qwen2): rotate split-halves (j, j + head_dim/2) rather than adjacent
// pairs. Same cos/sin cache as rope_apply; only the pairing differs. Bit-identical
// to the CPU rope_neox branch (separate-rounding mul/sub via volatile).
kernel void rope_apply_neox(
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
    uint hd = head_dim / 2u;
    uint head_off = gid * head_dim;
    for (uint j = 0u; j < hd; j++) {
        float a = x[head_off + j];
        float b = x[head_off + j + hd];
        float c = cosc[s * hd + j];
        float sn = sinc[s * hd + j];
        volatile float p1 = a * c; volatile float p2 = b * sn;
        out[head_off + j] = p1 - p2;
        volatile float p3 = a * sn; volatile float p4 = b * c;
        out[head_off + j + hd] = p3 + p4;
    }
}

// Multi-head attention with KV cache, one thread per query head. Matches the
// CPU forward's CANONICAL reductions (regime-2): Q.K via canonical_dot over
// head_size, /sqrt(head_size), canonical-sum softmax, and A.V as a canonical
// reduction over time per output dim. GQA: query head h reads kv head h/kv_mul.
// Caps context at 512 positions (< CANON_CHUNK=8192, so single-chunk canonical
// is bit-exact; raise the tile + add chunking for longer context).
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
    float sqrt_hs = sqrt((float)head_size);
    uint tlen = pos + 1u;
    // Q.K: canonical_dot over head_size (head_size < CANON_CHUNK).
    for (uint t = 0u; t < tlen; t++) {
        uint koff = t * kv_dim + kvhead * head_size;
        float lane[8] = {0,0,0,0,0,0,0,0};
        uint full = head_size - (head_size % 8u);
        uint dd = 0u;
        for (; dd < full; dd += 8u) {
            for (uint j = 0u; j < 8u; j++) {
                float p = q[qoff + dd + j] * kcache[koff + dd + j];
                lane[j] += p;
            }
        }
        for (; dd < head_size; dd++) { uint j = dd % 8u; float p = q[qoff + dd] * kcache[koff + dd]; lane[j] += p; }
        scores[t] = div_sw(ftree8(lane), sqrt_hs); // score / sqrt(head_size)
    }
    // softmax over time: max, exp, canonical-sum denominator (== softmax_inplace).
    float mx = -INFINITY;
    for (uint t = 0u; t < tlen; t++) { if (scores[t] > mx) mx = scores[t]; }
    for (uint t = 0u; t < tlen; t++) { scores[t] = vt_expf(scores[t] - mx); }
    {
        float lane[8] = {0,0,0,0,0,0,0,0};
        uint full = tlen - (tlen % 8u);
        uint t = 0u;
        for (; t < full; t += 8u) { for (uint j = 0u; j < 8u; j++) lane[j] += scores[t + j]; }
        for (; t < tlen; t++) lane[t % 8u] += scores[t];
        float inv = div_sw(1.0f, ftree8(lane));
        for (uint k = 0u; k < tlen; k++) scores[k] = scores[k] * inv;
    }
    // A.V: for each output dim, canonical_dot over time of scores[t]·V[t][d]
    // (value column strided by kv_dim). Written once per (h,d) — assign, not +=.
    for (uint dd = 0u; dd < head_size; dd++) {
        float lane[8] = {0,0,0,0,0,0,0,0};
        uint full = tlen - (tlen % 8u);
        uint t = 0u;
        for (; t < full; t += 8u) {
            for (uint j = 0u; j < 8u; j++) {
                uint voff = (t + j) * kv_dim + kvhead * head_size;
                float p = scores[t + j] * vcache[voff + dd];
                lane[j] += p;
            }
        }
        for (; t < tlen; t++) {
            uint voff = t * kv_dim + kvhead * head_size;
            float p = scores[t] * vcache[voff + dd];
            lane[t % 8u] += p;
        }
        xbout[qoff + dd] = ftree8(lane);
    }
}

// Opt-in unchanged-profile attention. One group per query head; independent
// Q.K positions, exponentials and A.V columns are distributed across 64 threads.
// Max retains its original serial comparison order. Denominator threads retain
// the original eight chains and fixed tree; no padding or reassociation.
kernel void attention_cooperative(
    device const float* q [[buffer(0)]], device const float* kcache [[buffer(1)]],
    device const float* vcache [[buffer(2)]], device float* xbout [[buffer(3)]],
    constant uint* d [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp contract(off)
    uint h = group.x;
    uint n_heads=d[0], head_size=d[1], kv_dim=d[2], kv_mul=d[3], pos=d[4];
    if (h>=n_heads || pos>=512u || head_size==0u || head_size>8192u || kv_mul==0u) return;
    uint tlen=pos+1u, qoff=h*head_size, kvhead=h/kv_mul;
    float sqrt_hs=sqrt(float(head_size));
    threadgroup float scores[512];
    threadgroup float sums[8];
    threadgroup float normalization[2];
    for (uint t=tid;t<tlen;t+=64u) {
        uint koff=t*kv_dim+kvhead*head_size;
        float lane[8]={0,0,0,0,0,0,0,0};
        uint full=head_size-(head_size%8u);
        uint dd=0u;
        for (;dd<full;dd+=8u) {
            for (uint j=0u;j<8u;j++) {
                float p=q[qoff+dd+j]*kcache[koff+dd+j];
                lane[j]+=p;
            }
        }
        for (;dd<head_size;dd++) {
            float p=q[qoff+dd]*kcache[koff+dd];
            lane[dd%8u]+=p;
        }
        scores[t]=div_sw(ftree8(lane),sqrt_hs);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0u) {
        float mx=-INFINITY;
        for (uint t=0u;t<tlen;t++) if (scores[t]>mx) mx=scores[t];
        normalization[0]=mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t=tid;t<tlen;t+=64u) scores[t]=vt_expf(scores[t]-normalization[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<8u) {
        float sum=0.0f;
        for (uint t=tid;t<tlen;t+=8u) sum+=scores[t];
        sums[tid]=sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0u) {
        float lane[8];
        for (uint j=0u;j<8u;j++) lane[j]=sums[j];
        normalization[1]=div_sw(1.0f,ftree8(lane));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t=tid;t<tlen;t+=64u) scores[t]=scores[t]*normalization[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dd=tid;dd<head_size;dd+=64u) {
        float lane[8]={0,0,0,0,0,0,0,0};
        uint full=tlen-(tlen%8u), t=0u;
        for (;t<full;t+=8u) {
            for (uint j=0u;j<8u;j++) {
                uint voff=(t+j)*kv_dim+kvhead*head_size;
                float p=scores[t+j]*vcache[voff+dd];
                lane[j]+=p;
            }
        }
        for (;t<tlen;t++) {
            float p=scores[t]*vcache[t*kv_dim+kvhead*head_size+dd];
            lane[t%8u]+=p;
        }
        xbout[qoff+dd]=ftree8(lane);
    }
}

// SiLU keeps the CPU's x/(1+exp(-x)) rounding, including its overflow-to-zero
// behavior. Scale both operands exactly when the reciprocal would be subnormal.
static inline float vt_silu(float x) {
    uint word = as_type<uint>(x);
    uint mag = word & 0x7fffffffu;
    if (mag < 0x01000000u) {
        // exp(-x) rounds to 1 here, so the denominator is exactly 2. A bitwise
        // half preserves subnormal results even on a flush-to-zero device.
        uint halved = (mag >> 1) + ((mag & 3u) == 3u ? 1u : 0u);
        return as_type<float>((word & 0x80000000u) | halved);
    }
    float denominator = 1.0f + vt_expf(-x);
    uint denom = as_type<uint>(denominator);
    if (denom >= 0x7b800000u && mag < 0x7f800000u) { // denominator >= 2^120
        if (denom == 0x7f800000u) return as_type<float>(word & 0x80000000u);
        return div_sw(x * 0x1p-64f, denominator * 0x1p-64f);
    }
    return div_sw(x, denominator);
}

static inline ulong vt_round_shift(ulong value, int shift) {
    if (shift <= 0) return value << uint(-shift);
    if (shift >= 64) return 0ul;
    ulong integral = value >> uint(shift);
    ulong residual = value & ((1ul << uint(shift)) - 1ul);
    ulong halfway = 1ul << uint(shift - 1);
    return integral + ulong(residual > halfway || (residual == halfway && (integral & 1ul)));
}

// Ordinary normal products stay on the hardware path. Only inputs/results in
// the underflow range use the exact 24x24-bit product and ties-to-even packing.
static inline float vt_silu_product(float a, float b) {
    uint wa = as_type<uint>(a), wb = as_type<uint>(b);
    uint ma = wa & 0x7fffffffu, mb = wb & 0x7fffffffu;
    uint ea = ma >> 23, eb = mb >> 23;
    if ((ea != 0u && eb != 0u && ea + eb >= 128u) || ea == 255u || eb == 255u)
        return a * b;
    uint sign = (wa ^ wb) & 0x80000000u;
    uint sa = (ma & 0x7fffffu) | (ea ? 0x800000u : 0u);
    uint sb = (mb & 0x7fffffu) | (eb ? 0x800000u : 0u);
    ulong product = ulong(sa) * ulong(sb);
    if (product == 0ul) return as_type<float>(sign);
    int exponent = (ea ? int(ea) - 150 : -149) + (eb ? int(eb) - 150 : -149);
    int top = 63 - int(clz(product));
    int unbiased = top + exponent;
    if (unbiased < -126) {
        uint packed = uint(vt_round_shift(product, -149 - exponent));
        return as_type<float>(sign | packed);
    }
    ulong significand = vt_round_shift(product, top - 23);
    if (significand == 0x1000000ul) { significand >>= 1; unbiased++; }
    if (unbiased > 127) return as_type<float>(sign | 0x7f800000u);
    return as_type<float>(sign | (uint(unbiased + 127) << 23) | (uint(significand) & 0x7fffffu));
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
    out[gid] = vt_silu_product(vt_silu(g), up[gid]);
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
    out[gid] = vt_silu(v);
}

// Opt-in local contraction-off RMS: unchanged operations, no volatile intermediates.
kernel void rms_kernel_no_contract(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    constant     float& eps  [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp contract(off)
    uint feat = dims[0], rows = dims[1], r = group.x;
    // Uniform across the entire group, so this cannot strand a barrier.
    if (r >= rows) return;
    device const float* row = x + (ulong)r * feat;
    threadgroup float partials[8];
    threadgroup float shared_scale;
    if (tid < 8u) {
        float sum = 0.0f;
        for (uint i = tid; i < feat; i += 8u) {
            float product = row[i] * row[i];
            sum += product;
        }
        partials[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float lanes[8];
        for (uint j = 0u; j < 8u; j++) lanes[j] = partials[j];
        float sumsq = ftree8(lanes);
        float mean = div_sw(sumsq, (float)feat);
        shared_scale = div_sw(1.0f, sqrt(mean + eps));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared_scale;
    device float* orow = out + (ulong)r * feat;
    for (uint i = tid; i < feat; i += 64u) {
        float scaled = row[i] * scale;
        orow[i] = scaled * w[i];
    }
}
