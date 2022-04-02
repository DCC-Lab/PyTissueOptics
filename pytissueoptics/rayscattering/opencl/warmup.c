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

__kernel void moveBy(__global photonStruct *a, float step_size)
{
    int gid = get_global_id(0);
        a[gid].position += step_size * a[gid].direction;

}

__kernel void decreaseWeightBy(__global photonStruct *a, float delta_weight)
{
    int gid = get_global_id(0);
        a[gid].weight -= delta_weight;
}

__kernel void propagate(__global photonStruct *a)
{
    int gid = get_global_id(0);
    while (a[gid].weight > 0.0f)
    {
        moveBy(a, 1.0f);
        decreaseWeightBy(a, 0.01f);
    }

}
//__kernel void roulette(__global photonStruct *a, __global float *rdn)
//{
//    int gid = get_global_id(0);
//    float chance = 0.1;
//    if (a[gid].weight >= 1e-4 || a[gid].weight == 0){return;}
//    else if (rnd[gid] < chance){a[gid].weight /= chance;}
//    else{a[gid].weight = 0;}
//}