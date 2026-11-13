// Each eight-thread subgroup owns two independent output rows. Per-row
// canonical lanes, rounded operations, chunk boundaries and trees are unchanged.
// Compile with VITNI_Q4K_NO_CONTRACT=1, as in the retained metadata baseline.
static inline float2 q4k_metadata_two_row_dot(
    device const float* x, device const uchar* w, uint K, uint M,
    uint nsuper, uint gid, uint tid, threadgroup float* partials)
{
    uint row0 = (gid / CANON_LANES) * 2u;
    uint row1 = row0 + 1u;
    uint lane = tid % CANON_LANES;
    uint row_slot = tid - lane;
    bool active0 = row0 < M;
    bool active1 = row1 < M;
    float sum0 = 0.0f, sum1 = 0.0f;
    float chunks0[64], chunks1[64];
    uint nchunks = 0u;
    uint supers_per_chunk = CANON_CHUNK / Q4K_NUMEL;
    for (uint b = 0u; b < nsuper; b++) {
        uint base = b * Q4K_NUMEL;
        if (base >= K) break;
        if (active0) {
            device const uchar* blk0 = w + ((ulong)row0 * nsuper + b) * 192u;
            // A missing second row never addresses an out-of-range block.
            device const uchar* blk1 = w + ((ulong)(active1 ? row1 : row0) * nsuper + b) * 192u;
            device const float* scale0 = (device const float*)blk0;
            device const float* scale1 = (device const float*)blk1;
            device const uchar* qs0 = blk0 + 64u;
            device const uchar* qs1 = blk1 + 64u;
            for (uint sub = 0u; sub < 4u; sub++) {
                float d10 = scale0[4u*sub], m10 = scale0[4u*sub+1u];
                float d20 = scale0[4u*sub+2u], m20 = scale0[4u*sub+3u];
                float d11 = scale1[4u*sub], m11 = scale1[4u*sub+1u];
                float d21 = scale1[4u*sub+2u], m21 = scale1[4u*sub+3u];
                // Finish the low half before high, exactly as each original lane.
                for (uint t = lane; t < 32u; t += CANON_LANES) {
                    uint i = base + 64u*sub + t;
                    if (i < K) {
                        float xi = x[i];
                        Q4K_ROUNDED float dq0 = d10 * (float)(qs0[32u*sub+t] & 0x0fu);
                        float weight0 = dq0 - m10;
                        Q4K_ROUNDED float product0 = xi * weight0;
                        if (active1) {
                            Q4K_ROUNDED float dq1 = d11 * (float)(qs1[32u*sub+t] & 0x0fu);
                            float weight1 = dq1 - m11;
                            Q4K_ROUNDED float product1 = xi * weight1;
                            sum1 += product1;
                        }
                        sum0 += product0;
                    }
                }
                for (uint t = lane; t < 32u; t += CANON_LANES) {
                    uint i = base + 64u*sub + 32u + t;
                    if (i < K) {
                        float xi = x[i];
                        Q4K_ROUNDED float dq0 = d20 * (float)(qs0[32u*sub+t] >> 4);
                        float weight0 = dq0 - m20;
                        Q4K_ROUNDED float product0 = xi * weight0;
                        if (active1) {
                            Q4K_ROUNDED float dq1 = d21 * (float)(qs1[32u*sub+t] >> 4);
                            float weight1 = dq1 - m21;
                            Q4K_ROUNDED float product1 = xi * weight1;
                            sum1 += product1;
                        }
                        sum0 += product0;
                    }
                }
            }
        }
        bool is_last = base + min(Q4K_NUMEL, K-base) >= K;
        if (((b+1u) % supers_per_chunk == 0u) || is_last) {
            partials[tid] = sum0;
            partials[64u+tid] = sum1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                float lanes0[8], lanes1[8];
                for (uint j=0u; j<CANON_LANES; j++) {
                    lanes0[j] = partials[row_slot+j];
                    lanes1[j] = partials[64u+row_slot+j];
                }
                chunks0[nchunks] = fixed_tree(lanes0, CANON_LANES);
                chunks1[nchunks] = fixed_tree(lanes1, CANON_LANES);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            nchunks++;
            sum0=0.0f; sum1=0.0f;
        }
    }
    if (lane == 0u) return float2(fixed_tree(chunks0,nchunks),fixed_tree(chunks1,nchunks));
    return float2(0.0f);
}

kernel void q4k_linear_metadata_two_row(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[128];
    float2 dot=q4k_metadata_two_row_dot(x,w,dims[0],dims[1],dims[2],gid,tid,partials);
    uint row=(gid/CANON_LANES)*2u;
    if (tid%CANON_LANES==0u) {
        if (row<dims[1]) out[row]=dot.x;
        if (row+1u<dims[1]) out[row+1u]=dot.y;
    }
}

kernel void q4k_linear_metadata_two_row_acc(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[128];
    float2 dot=q4k_metadata_two_row_dot(x,w,dims[0],dims[1],dims[2],gid,tid,partials);
    uint row=(gid/CANON_LANES)*2u;
    if (tid%CANON_LANES==0u) {
        if (row<dims[1]) out[row]=out[row]+dot.x;
        if (row+1u<dims[1]) out[row+1u]=out[row+1u]+dot.y;
    }
}
