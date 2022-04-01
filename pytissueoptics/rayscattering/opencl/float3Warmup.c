//------------------------------------------------------------------------------
//
// kernel:  pi
//
// Purpose: accumulate partial sums of pi comp
//
// input: float step_size
//        int   niters per work item
//        local float* an array to hold sums from each work item
//
// output: partial_sums   float vector of partial sums
//


__kernel void vectorDouble(__global float3 *a)
{
    int gid = get_global_id(0);
    a[gid] += a[gid];
}