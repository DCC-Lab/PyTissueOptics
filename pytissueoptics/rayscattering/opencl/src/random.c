uint wangHash(uint seed){
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float getRandomFloatValue(__global unsigned int *seedBuffer, unsigned int id){
     float result = 0.0f;
     while(result == 0.0f){
         uint rnd_seed = wangHash(seedBuffer[id]);
         seedBuffer[id] = rnd_seed;
         result = (float)rnd_seed / (float)UINT_MAX;
     }
     return result;
    }

 __kernel void fillRandomFloatBuffer(__global unsigned int *seedBuffer, __global float *randomFloatBuffer){
    int id = get_global_id(0);
    randomFloatBuffer[id] = getRandomFloatValue(seedBuffer, id);
    }
