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


__kernel void moveByOne(__global photonStruct *a)
{
    int ix = get_global_id(0);
    a[ix].position += a[ix].direction;
}