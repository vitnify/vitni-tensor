// Research only: 32 product producers, original eight ordered accumulators.
// Append to contraction-controlled q4k.metal; host requires SIMD width 32.
static inline float q4k_product_prepare_dot(
    device const float* x, device const uchar* w, uint K, uint M,
    uint nsuper, uint gid, uint tid, threadgroup float* partials)
{
    uint row = gid / 32u;
    uint physical = tid % 32u;
    uint reducer = physical % 8u;
    uint slot = (tid / 32u) * 8u;
    bool active = row < M;
    float sum = 0.0f;
    float chunks[64];
    uint nchunks = 0u;
    for (uint b = 0u; b < nsuper; ++b) {
        uint base = b * 256u;
        if (base >= K) break;
        for (uint sub = 0u; sub < 4u; ++sub) {
            for (uint segment = 0u; segment < 2u; ++segment) {
                uint start = base + sub * 64u + segment * 32u;
                float product = 0.0f;
                if (active && start + physical < K) {
                    device const uchar* block = w + ((ulong)row * nsuper + b) * 192u;
                    device const float* sm = (device const float*)block;
                    uint q = block[64u + sub * 32u + physical];
                    uint nibble = segment == 0u ? (q & 15u) : (q >> 4u);
                    float dq = sm[4u * sub + 2u * segment] * (float)nibble;
                    float weight = dq - sm[4u * sub + 2u * segment + 1u];
                    product = x[start + physical] * weight;
                }
                // All 32 source lanes execute every shuffle together. Inactive
                // rows still participate; missing tail products are not added.
                for (uint j = 0u; j < 4u; ++j) {
                    float p = simd_shuffle(product, (ushort)(reducer + 8u * j));
                    if (physical < 8u && active && start + reducer + 8u * j < K)
                        sum += p;
                }
            }
        }
        bool last = base + min(256u, K - base) >= K;
        if ((b + 1u) % 32u == 0u || last) {
            if (physical < 8u) partials[slot + physical] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (physical == 0u) {
                float lanes[8];
                for (uint j = 0u; j < 8u; ++j) lanes[j] = partials[slot + j];
                chunks[nchunks] = fixed_tree(lanes, 8u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ++nchunks;
            sum = 0.0f;
        }
    }
    return physical == 0u ? fixed_tree(chunks, nchunks) : 0.0f;
}

kernel void q4k_linear_product_prepare(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[64];
    float dot = q4k_product_prepare_dot(x, w, dims[0], dims[1], dims[2], gid, tid, partials);
    if (tid % 32u == 0u && gid / 32u < dims[1]) out[gid / 32u] = dot;
}

kernel void q4k_linear_product_prepare_acc(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[64];
    float dot = q4k_product_prepare_dot(x, w, dims[0], dims[1], dims[2], gid, tid, partials);
    if (tid % 32u == 0u && gid / 32u < dims[1]) out[gid / 32u] = out[gid / 32u] + dot;
}
