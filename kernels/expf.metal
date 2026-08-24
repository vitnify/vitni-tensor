#include <metal_stdlib>
using namespace metal;

static inline float div_sw(float a, float b); // cross-vendor software divide (defined below)

// ===================================================================
// expf — bit-exact port of libm 0.2.16 `expf` (musl e_expf), the exact
// function vitni-tensor calls in softmax and SiLU. The whole point: the
// hot-path transcendental is f32-only (no f64), so it reproduces on Apple
// GPUs (Metal has no f64 at all). Compile with fastMathEnabled = false so
// the polynomial's multiplies and adds stay separately rounded (no fma) —
// same numerical contract as the matmul reduction.
// ===================================================================

// scalbnf(x, n) = x * 2^n, exact port of libm generic scalbn for f32
// (EXP_MAX=127, EXP_MIN=-126, sig_total_bits=24, EXP_BIAS=127).
static inline float vt_scalbnf(float x, int n) {
    float f_exp_max     = as_type<float>((uint)254u << 23); // 2^127
    float f_exp_min     = as_type<float>((uint)1u   << 23); // 2^-126
    float f_pow_subnorm = as_type<float>((uint)151u << 23); // 2^24
    if (n > 127) {
        x = x * f_exp_max; n -= 127;
        if (n > 127) {
            x = x * f_exp_max; n -= 127;
            if (n > 127) n = 127;
        }
    } else if (n < -126) {
        float mul = f_exp_min * f_pow_subnorm; // 2^-102
        int add = 126 - 24;                    // -EXP_MIN - sig_total_bits = 102
        x = x * mul; n += add;
        if (n < -126) {
            x = x * mul; n += add;
            if (n < -126) n = -126;
        }
    }
    float scale = as_type<float>((uint)((127 + n)) << 23);
    return x * scale;
}

static inline float vt_expf(float x) {
    const float HALF0 = 0.5f, HALF1 = -0.5f;
    const float LN2_HI  = 6.9314575195e-01f;
    const float LN2_LO  = 1.4286067653e-06f;
    const float INV_LN2 = 1.4426950216e+00f;
    const float P1 = 1.6666625440e-1f;
    const float P2 = -2.7667332906e-3f;

    float x1p127  = as_type<float>((uint)0x7f000000u); // 2^127
    uint hx = as_type<uint>(x);
    int sign = (int)(hx >> 31);
    bool signb = sign != 0;
    hx &= 0x7fffffffu; // |x| bits

    // special cases
    if (hx >= 0x42aeac50u) {           // |x| >= 87.33655 or NaN
        if (hx > 0x7f800000u) return x; // NaN
        if ((hx >= 0x42b17218u) && !signb) { // x >= 88.722839 -> overflow
            x = x * x1p127;
            return x;
        }
        if (signb) {                    // underflow region
            if (hx >= 0x42cff1b5u) return 0.0f; // x <= -103.972084 -> 0
        }
    }

    // argument reduction
    int k;
    float hi, lo;
    if (hx > 0x3eb17218u) {             // |x| > 0.5 ln2
        if (hx > 0x3f851592u) {         // |x| > 1.5 ln2
            float h = signb ? HALF1 : HALF0;
            volatile float ix = INV_LN2 * x; // block fma contraction of a*b+c
            k = (int)(ix + h);
        } else {
            k = 1 - sign - sign;
        }
        float kf = (float)k;
        volatile float khi = kf * LN2_HI; // block a-b*c contraction
        hi = x - khi;                   // k*ln2hi exact here
        lo = kf * LN2_LO;
        x = hi - lo;
    } else if (hx > 0x39000000u) {      // |x| > 2^-14
        k = 0; hi = x; lo = 0.0f;
    } else {
        return 1.0f + x;               // tiny x
    }

    // primary range polynomial. Each product is forced to round to f32 before
    // the add/sub (volatile) so the GPU cannot contract into an fma — matching
    // the CPU's separately-rounded ops, which is what makes it cross-vendor.
    float xx = x * x;
    volatile float m1 = xx * P2;
    float t = P1 + m1;
    volatile float m2 = xx * t;
    float c = x - m2;
    float xc = x * c;
    float y = 1.0f + ((div_sw(xc, 2.0f - c) - lo) + hi);
    return (k == 0) ? y : vt_scalbnf(y, k);
}

// Software f32 division using ONLY ops that are bit-identical across CPU / Metal
// / CUDA: integer bit-ops + correctly-rounded FMA. No hardware divide, no
// hardware reciprocal. Seed via an integer reciprocal magic, refine with
// Newton (fma), correct the quotient (fma). Same source runs on every device,
// so the RESULT is identical on every device by construction.
static inline float div_sw(float a, float b) {
    uint sgn = (as_type<uint>(a) ^ as_type<uint>(b)) & 0x80000000u;
    float ba = as_type<float>(as_type<uint>(b) & 0x7fffffffu); // |b|
    float aa = as_type<float>(as_type<uint>(a) & 0x7fffffffu); // |a|
    uint j = 0x7EF127EAu - as_type<uint>(ba);                  // 1/|b| seed
    float y = as_type<float>(j);
    float nb = -ba;
    float e;
    e = fma(nb, y, 1.0f); y = fma(y, e, y); // Newton step 1
    e = fma(nb, y, 1.0f); y = fma(y, e, y); // step 2
    e = fma(nb, y, 1.0f); y = fma(y, e, y); // step 3
    float q = aa * y;
    float r = fma(nb, q, aa);
    q = fma(r, y, q);                        // corrected quotient
    return as_type<float>(as_type<uint>(q) ^ sgn);
}

kernel void divsw_kernel(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = div_sw(a[gid], b[gid]);
}

// Isolation: the expf polynomial c = x - xx*(P1 + xx*P2). If GPU != CPU here,
// the optimizer is contracting the `x - xx*t` into an fnms despite fast-math off.
kernel void poly_kernel(
    device const float *in  [[buffer(0)]],
    device       float *out [[buffer(1)]],
    constant     uint  &n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    const float P1 = 1.6666625440e-1f;
    const float P2 = -2.7667332906e-3f;
    float x = in[gid];
    float xx = x * x;
    // Force each product to round to f32 before the add/sub, blocking the
    // optimizer from contracting `a - b*c` into an fnms (which fast-math-off
    // does NOT prevent on Metal).
    volatile float m1 = xx * P2;
    float t = P1 + m1;
    volatile float m2 = xx * t;
    out[gid] = x - m2;
}

// Diagnostic: is Metal's f32 division correctly-rounded (fast-math off)?
kernel void div_kernel(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = a[gid] / b[gid];
}

kernel void expf_kernel(
    device const float *in  [[buffer(0)]],
    device       float *out [[buffer(1)]],
    constant     uint  &n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = vt_expf(in[gid]);
}
