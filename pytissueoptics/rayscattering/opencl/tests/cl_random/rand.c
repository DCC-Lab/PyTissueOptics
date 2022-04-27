uint wangHash(uint seed){
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float getRandomFloatValue(__global unsigned int *seedBuffer, unsigned int id){
     uint maxint = 0;
     maxint--;
     uint rnd_int = wangHash(seedBuffer[id]);
     seedBuffer[id] = rnd_int;
     return ((float)rnd_int) / (float)maxint;
    }

 __kernel void fillRandomFloatBuffer(__global unsigned int *seedBuffer, __global float *randomFloatBuffer){
    int id = get_global_id(0);
    randomFloatBuffer[id] = getRandomFloatValue(seedBuffer, id);
    }