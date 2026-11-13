// Research-only finished-weight tables; canonical product/addition schedule unchanged.
static inline float q4k_lookup_dot(
    device const float* x, device const uchar* w, uint K, uint M,
    uint nsuper, uint gid, uint tid, threadgroup float* partials)
{
    uint row = gid / CANON_LANES;
    uint lane = tid % CANON_LANES;
    uint row_slot = tid - lane;
    bool active = row < M;
    float lane_sum = 0.0f;
    float chunk_sums[64];
    uint nchunks = 0u;
    uint supers_per_chunk = CANON_CHUNK / Q4K_NUMEL;

    for (uint b = 0u; b < nsuper; b++) {
        uint base = b * Q4K_NUMEL;
        if (base >= K) break;
        if (active) {
            device const uchar* blk = w + ((ulong)row * nsuper + b) * 640u;
            device const float* palette = (device const float*)blk;
            device const uchar* qs = blk + 512u;
            for (uint sub = 0u; sub < 4u; sub++) {
                // Low half must finish BEFORE high half: interleaving the two
                // would change each lane's floating-point addition sequence.
                for (uint t = lane; t < 32u; t += CANON_LANES) {
                    uint i = base + 64u * sub + t;
                    if (i < K) {
                        float weight = palette[32u * sub + (uint)(qs[32u * sub + t] & 0x0Fu)];
                        Q4K_ROUNDED float product = x[i] * weight;
                        lane_sum += product;
                    }
                }
                for (uint t = lane; t < 32u; t += CANON_LANES) {
                    uint i = base + 64u * sub + 32u + t;
                    if (i < K) {
                        float weight = palette[32u * sub + 16u + (uint)(qs[32u * sub + t] >> 4)];
                        Q4K_ROUNDED float product = x[i] * weight;
                        lane_sum += product;
                    }
                }
            }
        }
        bool is_last = base + min(Q4K_NUMEL, K - base) >= K;
        if (((b + 1u) % supers_per_chunk == 0u) || is_last) {
            partials[tid] = lane_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                float lanes[8];
                for (uint j = 0u; j < CANON_LANES; j++) lanes[j] = partials[row_slot + j];
                chunk_sums[nchunks] = fixed_tree(lanes, CANON_LANES);
            }
            // Every reader finishes before another chunk can overwrite parts.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            nchunks++;
            lane_sum = 0.0f;
        }
    }
    return lane == 0u ? fixed_tree(chunk_sums, nchunks) : 0.0f;
}

kernel void q4k_linear_lookup(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[64];
    float dot = q4k_lookup_dot(x, w, dims[0], dims[1], dims[2], gid, tid, partials);
    uint row = gid / CANON_LANES;
    if (tid % CANON_LANES == 0u && row < dims[1]) out[row] = dot;
}

kernel void q4k_linear_lookup_acc(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[64];
    float dot = q4k_lookup_dot(x, w, dims[0], dims[1], dims[2], gid, tid, partials);
    uint row = gid / CANON_LANES;
    if (tid % CANON_LANES == 0u && row < dims[1]) out[row] = out[row] + dot;
}
