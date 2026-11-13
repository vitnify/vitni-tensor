// Research fixture: four integer lanes per block, eight blocks per SIMD row.
// Valid-domain bound for every signed integer sum, including reassociation:
// 256 * 32 * 127 * 128 = 133169152 < INT32_MAX. Quantization is unchanged.
static inline int q6k_within_block_partial(device const char* qs,
    device const uchar* blk, uint ilane) {
    uint hb = ilane / 2u, pass = ilane % 2u;
    uint ql_p = hb * 64u, qh_p = hb * 32u;
    uint sc_p = hb * 8u + pass, y = hb * 128u;
    int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    for (uint l = pass * 16u; l < pass * 16u + 16u; l++) {
        uint h = (uint)blk[128u + qh_p + l];
        uint b0 = (uint)blk[ql_p + l], b1 = (uint)blk[ql_p + l + 32u];
        a1 += ((int)((b0 & 0x0Fu) | ((h & 0x03u) << 4)) - 32) * (int)qs[y + l];
        a2 += ((int)((b1 & 0x0Fu) | (((h >> 2) & 0x03u) << 4)) - 32) * (int)qs[y + l + 32u];
        a3 += ((int)((b0 >> 4) | (((h >> 4) & 0x03u) << 4)) - 32) * (int)qs[y + l + 64u];
        a4 += ((int)((b1 >> 4) | (((h >> 6) & 0x03u) << 4)) - 32) * (int)qs[y + l + 96u];
    }
    return (int)((char)blk[192u + sc_p]) * a1
         + (int)((char)blk[192u + sc_p + 2u]) * a2
         + (int)((char)blk[192u + sc_p + 4u]) * a3
         + (int)((char)blk[192u + sc_p + 6u]) * a4;
}

static inline float q6k_within_block_dot(device const float* xdx,
    device const char* xqs, device const uchar* w, uint M, uint nsuper,
    uint gid, uint tid) {
#pragma METAL fp contract(off)
    uint row = gid / 32u, simd_tid = tid % 32u;
    uint blane = simd_tid / 4u, ilane = simd_tid % 4u;
    float roots[8];
    uint nchunks = 0u;
    for (uint done = 0u; done < nsuper; done += 8u) {
        uint len = min(8u, nsuper - done), s = done + blane;
        bool active = row < M && blane < len;
        int partial = 0;
        if (active) {
            device const uchar* blk = w + ((ulong)row * nsuper + s) * Q6K_BYTES;
            partial = q6k_within_block_partial(xqs + s * 256u, blk, ilane);
        }
        // All SIMD lanes participate, including invalid row/block tails.
        int acc = partial + simd_shuffle_xor(partial, 1u);
        acc += simd_shuffle_xor(acc, 2u);
        float value = 0.0f;
        if (active && ilane == 0u) {
            device const uchar* blk = w + ((ulong)row * nsuper + s) * Q6K_BYTES;
            float dw = f16_to_f32((ushort)((uint)blk[208] | ((uint)blk[209] << 8)));
            // Same two f32 multiplications and rounding boundaries as baseline.
            value = dw * xdx[s] * (float)acc;
        }
        // Original adjacent-pair float tree, reading one value every four lanes.
        uint count = len;
        while (count > 1u) {
            uint half_count = (count + 1u) / 2u, u = 2u * blane;
            float left = simd_shuffle(value, 4u * (u & 7u));
            float right = simd_shuffle(value, 4u * ((u + 1u) & 7u));
            if (ilane == 0u && blane < half_count)
                value = u + 1u < count ? left + right : left;
            count = half_count;
        }
        if (simd_tid == 0u) roots[nchunks] = value;
        nchunks++;
    }
    return simd_tid == 0u ? fixed_tree(roots, nchunks) : 0.0f;
}

kernel void q6k_integer_dot_within_block(
    device const float* xdx [[buffer(0)]], device const char* xqs [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (dims[2] > 64u) return;
    float dot = q6k_within_block_dot(xdx, xqs, w, dims[1], dims[2], gid, tid);
    uint row = gid / 32u;
    if (tid % 32u == 0u && row < dims[1]) out[row] = dot;
}

kernel void q6k_integer_dot_within_block_acc(
    device const float* xdx [[buffer(0)]], device const char* xqs [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (dims[2] > 64u) return;
    float dot = q6k_within_block_dot(xdx, xqs, w, dims[1], dims[2], gid, tid);
    uint row = gid / 32u;
    if (tid % 32u == 0u && row < dims[1]) out[row] = out[row] + dot;
}
