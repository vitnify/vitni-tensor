static inline float rms_q4_metadata_dot(
    threadgroup const float* x, device const uchar* w, uint K, uint M,
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
            device const uchar* blk = w + ((ulong)row * nsuper + b) * 192u;
            device const float* scale_min = (device const float*)blk;
            device const uchar* qs = blk + 64u;
            for (uint sub = 0u; sub < 4u; sub++) {
                float d1 = scale_min[4u * sub];
                float m1f = scale_min[4u * sub + 1u];
                float d2 = scale_min[4u * sub + 2u];
                float m2f = scale_min[4u * sub + 3u];
                // Low half must finish BEFORE high half: interleaving the two
                // would change each lane's floating-point addition sequence.
                for (uint t = lane; t < 32u; t += CANON_LANES) {
                    uint i = base + 64u * sub + t;
                    if (i < K) {
                        Q4K_ROUNDED float dq = d1 * (float)(qs[32u * sub + t] & 0x0Fu);
                        float weight = dq - m1f;
                        Q4K_ROUNDED float product = x[i] * weight;
                        lane_sum += product;
                    }
                }
                for (uint t = lane; t < 32u; t += CANON_LANES) {
                    uint i = base + 64u * sub + 32u + t;
                    if (i < K) {
                        Q4K_ROUNDED float dq = d2 * (float)(qs[32u * sub + t] >> 4);
                        float weight = dq - m2f;
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

// Appended to unchanged contraction-off Q4 and forward helpers, plus fusion_dot.
static inline float rms_q4_fused_value(device const float* x,
    device const float* rms_w, device const uchar* w, constant uint* dims,
    float eps, threadgroup float* normalized, threadgroup float* partials,
    threadgroup float* shared_scale, uint gid, uint tid) {
#pragma METAL fp contract(off)
    uint K = dims[0];
    if (tid < 8u) {
        float sum = 0.0f;
        for (uint i = tid; i < K; i += 8u) {
            float square = x[i] * x[i];
            sum += square;
        }
        partials[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float lanes[8];
        for (uint j = 0u; j < 8u; j++) lanes[j] = partials[j];
        float mean = div_sw(ftree8(lanes), (float)K);
        *shared_scale = div_sw(1.0f, sqrt(mean + eps));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = *shared_scale;
    for (uint i = tid; i < K; i += 64u) {
        float scaled = x[i] * scale;
        normalized[i] = scaled * rms_w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return rms_q4_metadata_dot(normalized, w, K, dims[1], dims[2], gid, tid, partials);
}

kernel void rms_q4_fused(
    device const float* x [[buffer(0)]], device const float* rms_w [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], constant float& eps [[buffer(5)]],
    threadgroup float* normalized [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (dims[0] == 0u || dims[0] > 4096u || dims[0] % 256u != 0u) return;
    threadgroup float partials[64];
    threadgroup float shared_scale;
    float value = rms_q4_fused_value(x, rms_w, w, dims, eps, normalized,
                                    partials, &shared_scale, gid, tid);
    uint row = gid / 8u;
    if (tid % 8u == 0u && row < dims[1]) out[row] = value;
}

kernel void rms_q4_fused_acc(
    device const float* x [[buffer(0)]], device const float* rms_w [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], constant float& eps [[buffer(5)]],
    threadgroup float* normalized [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (dims[0] == 0u || dims[0] > 4096u || dims[0] % 256u != 0u) return;
    threadgroup float partials[64];
    threadgroup float shared_scale;
    float value = rms_q4_fused_value(x, rms_w, w, dims, eps, normalized,
                                    partials, &shared_scale, gid, tid);
    uint row = gid / 8u;
    if (tid % 8u == 0u && row < dims[1]) out[row] = out[row] + value;
}
