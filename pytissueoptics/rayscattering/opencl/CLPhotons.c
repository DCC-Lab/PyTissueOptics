
__kernel void decreaseWeightBy(__global photonStruct *photon, float *delta_weight, __global loggerStruct *logger, int loggerIndex)
{
    int gid = get_global_id(0);
        photon[gid].weight -= delta_weight[gid];
        if photon[gid].weight > 0.0001
        {
            logger[loggerIndex + gid].position = photon[gid].position;
            logger[loggerIndex + gid].delta_weight = delta_weight[gid];
            loggerIndex++;
        }
}


__kernel void interact(__global photonStruct *photon, __constant materialStruct *mat)
{
    int gid = get_global_id(0);
    float delta = photon[gid].weight * mat[photon[gid].material_id].albedo;
    decreaseWeightBy(photon, delta);
}


__kernel void moveBy(__global photonStruct *photon, float *distance)
{
    int gid = get_global_id(0);
        photon[gid].position += distance[gid] * photon[gid].direction;
}


    



// MATERIAL KERNELS
__kernel void getScatteringDistances(__constant materialStruct *material, int amount)
{
    int gid = get_global_id(0);

}


// PROPAGATE KERNELS
__kernel void propagate(__global photonStruct *photons, __global float *randomNum, uint rnd_index)
{
    int gid = get_global_id(0);
    while(photons[gid].weight > 0)
    {
        moveBy(photons, 1.0f);
        decreaseWeightBy(photons, 0.3f);
    }
}





//__kernel void roulette(__global photonStruct *photon, ){
//    int gid = get_global_id(0);
//    uint2 seed;
//    seed.x = 1;
//    seed.y = 2;
//    float *randomBuffer;
//    float chance = 0.1;
//    if (photon[gid].weight >= 0.0001){return;}
//    else if (randomBuffer[gid] < chance){
//        photon[gid].weight /= chance;
//    }
//    else{
//        photon[gid].weight = 0;
//    }
