// Research-only product staging; all original addition chains retained.
kernel void rms_kernel_product_stage(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint*  dims [[buffer(3)]],
    constant     float& eps  [[buffer(4)]],
    threadgroup float* squares [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp contract(off)
    uint feat = dims[0], rows = dims[1], r = group.x;
    // Uniform across the entire group, so this cannot strand a barrier.
    if (r >= rows || feat == 0u || feat > 4096u) return;
    device const float* row = x + (ulong)r * feat;
    threadgroup float partials[8];
    threadgroup float shared_scale;
    for (uint i = tid; i < feat; i += 64u) squares[i] = row[i] * row[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        float sum = 0.0f;
        for (uint i = tid; i < feat; i += 8u) {
            sum += squares[i];
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
