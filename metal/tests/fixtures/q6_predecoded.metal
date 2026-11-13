// Research-only resident signed-int8 Q6 layout, original integer grouping and float tree.
static inline float q6k_predecoded_part(device const float* xdx,
    device const char* xqs, device const uchar* wrow, uint s) {
#pragma METAL fp contract(off)
    device const char* qs = xqs + s * 256u;
    float dx = xdx[s];
    device const uchar* blk = wrow + s * 276u;
    device const char* decoded = (device const char*)blk;
    device const uchar* sc = blk + 256;
    float d_w = f16_to_f32((ushort)((uint)blk[272] | ((uint)blk[273] << 8)));
    int sums[16];
    for (uint j = 0u; j < 16u; j++) sums[j] = 0;
    for (uint hb = 0u; hb < 2u; hb++) {
        uint ql_p = hb * 64u, qh_p = hb * 32u, sc_p = hb * 8u, y = hb * 128u;
        for (uint pass = 0u; pass < 2u; pass++) {
            uint is = pass, l0 = pass * 16u;
            int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
            for (uint l = l0; l < l0 + 16u; l++) {
                a1 += (int)decoded[y+l] * (int)qs[y+l];
                a2 += (int)decoded[y+l+32u] * (int)qs[y+l+32u];
                a3 += (int)decoded[y+l+64u] * (int)qs[y+l+64u];
                a4 += (int)decoded[y+l+96u] * (int)qs[y+l+96u];
            }
            sums[sc_p + is] += a1; sums[sc_p + is + 2u] += a2; sums[sc_p + is + 4u] += a3; sums[sc_p + is + 6u] += a4;
        }
    }
    int acc = 0; for (uint j = 0u; j < 16u; j++) acc += (int)((char)sc[j]) * sums[j];
    return d_w * dx * (float)acc;
}

static inline float q6k_predecoded_tree_eight(float value, uint len, uint lane, uint simd_tid) {
    uint base = simd_tid - lane;
    while (len > 1u) {
        uint hlen = (len + 1u) / 2u;
        uint u = 2u * lane;
        // All eight subgroup members participate in every shuffle, including
        // unused tail members. Only valid tree nodes read the resulting values.
        float left = simd_shuffle(value, base + (u & 7u));
        float right = simd_shuffle(value, base + ((u + 1u) & 7u));
        if (lane < hlen) value = u + 1u < len ? left + right : left;
        len = hlen;
    }
    return value;
}

static inline float q6k_predecoded_dot(device const float* xdx,
    device const char* xqs, device const uchar* w, uint M, uint nsuper,
    uint gid, uint tid) {
    uint row = gid / 8u;
    uint lane = tid % 8u;
    uint simd_tid = tid % 32u;
    float roots[8];
    uint nchunks = 0u;
    for (uint done = 0u; done < nsuper; done += 8u) {
        uint len = min(8u, nsuper - done);
        uint s = done + lane;
        float part = 0.0f;
        if (row < M && lane < len) {
            device const uchar* wrow = w + (ulong)row * nsuper * 276u;
            part = q6k_predecoded_part(xdx, xqs, wrow, s);
        }
        float root = q6k_predecoded_tree_eight(part, len, lane, simd_tid);
        if (lane == 0u) roots[nchunks] = root;
        nchunks++;
    }
    if (lane == 0u) return fixed_tree(roots, nchunks);
    return 0.0f;
}

kernel void q6k_integer_dot_predecoded(
    device const float* xdx [[buffer(0)]], device const char* xqs [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (dims[2] > 64u) return;
    float dot = q6k_predecoded_dot(xdx,xqs,w,dims[1],dims[2],gid,tid);
    uint row = gid / 8u;
    if (tid % 8u == 0u && row < dims[1]) out[row] = dot;
}

kernel void q6k_integer_dot_predecoded_acc(
    device const float* xdx [[buffer(0)]], device const char* xqs [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (dims[2] > 64u) return;
    float dot = q6k_predecoded_dot(xdx,xqs,w,dims[1],dims[2],gid,tid);
    uint row = gid / 8u;
    if (tid % 8u == 0u && row < dims[1]) out[row] = out[row] + dot;
}
