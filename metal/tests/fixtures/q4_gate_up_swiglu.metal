// Research only. Append after forward_ops.metal and contraction-off q4k.metal.
// Two independent eight-thread teams produce a paired gate/up coordinate.
static inline void q4_gate_up_body(
    device const float* x, device const uchar* wg, device const uchar* wu,
    device float* out, device float* gate_debug, device float* up_debug,
    constant uint* dims, uint gid, uint tid, threadgroup float* partials,
    threadgroup float* dots, bool debug)
{
    uint row = gid / 16u;
    uint team = (tid / 8u) % 2u;
    uint lane = tid % 8u;
    device const uchar* w = team == 0u ? wg : wu;
    float dot = q4k_metadata_dot(x, w, dims[0], dims[1], dims[2], row * 8u + lane, tid, partials);
    if (lane == 0u) dots[tid / 8u] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid % 16u == 0u && row < dims[1]) {
        float g = dots[tid / 8u];
        float u = dots[tid / 8u + 1u];
        if (debug) { gate_debug[row] = g; up_debug[row] = u; }
        out[row] = vt_silu_product(vt_silu(g), u);
    }
}

kernel void q4_gate_up_swiglu(
    device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[64];
    threadgroup float dots[8];
    // Debug stores are compiled out of this measured kernel.
    q4_gate_up_body(x, wg, wu, out, out, out, dims, gid, tid, partials, dots, false);
}

kernel void q4_gate_up_swiglu_debug(
    device const float* x [[buffer(0)]], device const uchar* wg [[buffer(1)]],
    device const uchar* wu [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]], device float* gate_debug [[buffer(5)]],
    device float* up_debug [[buffer(6)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[64];
    threadgroup float dots[8];
    q4_gate_up_body(x, wg, wu, out, gate_debug, up_debug, dims, gid, tid, partials, dots, true);
}
