// RANDOM
uint wang_hash(uint seed)
    {
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
    }


void randomize_buffer_seed(__global unsigned int * rnd_buffer, int id)
    {
     uint rnd_int = wang_hash(id);
     rnd_buffer[id] = rnd_int;
    }

float random_float(__global unsigned int * rnd_buffer, int id)
    {
     uint maxint = 0;
     maxint--;
     uint rnd_int = wang_hash(rnd_buffer[id]);
     rnd_buffer[id] = rnd_int;
     return ((float)rnd_int) / (float)maxint;
    }


 __kernel void randomize_seed_init(__global unsigned int *rnd_buffer)
    {
       int id = get_global_id(0);
       randomize_buffer_seed(rnd_buffer, id);
    }

 __kernel void fill_random_float_buffer(__global unsigned int * rnd_buffer, float * float_buffer)
    {
    int id = get_global_id(0);
    float_buffer[id] = random_float(rnd_buffer, id);
    }


// ------------------------------------------------------------------------------------------------
// PHOTON PHYSICS
// ------------------------------------------------------------------------------------------------


void decreaseWeightBy(__global photonStruct *photons, float delta_weight, uint gid)
{
        photons[gid].weight -= delta_weight;
}

void moveBy(__global photonStruct *photons, float distance, uint gid)
{
        photons[gid].position += distance * photons[gid].direction;
}

float getScatteringDistance(__global photonStruct *photons,__constant materialStruct *materials, __global float * randomNums, uint gid)
{
    return -log(randomNums[gid]) / materials[photons[gid].material_id].albedo;
}


// PROPAGATE KERNELS
__kernel void propagate(__global photonStruct *photons, __constant materialStruct *materials, __global float *randomNums, __global uint * rnd_buffer)
{
    int gid = get_global_id(0);
    while(photons[gid].weight > 0.0001)
    {
        randomNums[gid] = random_float(rnd_buffer, gid);
        float distance = getScatteringDistance(photons, materials, randomNums, gid);
        moveBy(photons, distance, gid);
        decreaseWeightBy(photons, 0.3f, gid);
    }
}
