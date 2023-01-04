uint wangHash(uint seed);

uint wangHash(uint seed){
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float getRandomFloatValue(__global unsigned int *seeds, unsigned int id){
     float result = 0.0f;
     while(result == 0.0f){
         uint rnd_seed = wangHash(seeds[id]);
         seeds[id] = rnd_seed;
         result = (float)rnd_seed / (float)UINT_MAX;
     }
     return result;
    }

 __kernel void fillRandomFloatBuffer(__global unsigned int *seeds, __global float *randomNumbers){
    int id = get_global_id(0);
    randomNumbers[id] = getRandomFloatValue(seeds, id);
    }
