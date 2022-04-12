
// This is a pseudo-random generator based on unsigned int hashing from Thomas Wang.
// The explanation and implementation of wang_hash were found on this website. All credits to Mr. Wang.
// http://web.archive.org/web/20071223173210/http://www.concentric.net/~Ttwang/tech/inthash.htm

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
    float_buffer[id] = wang_rnd(rnd_buffer, id);
    }


__kernel void testKernel(uint datasize, __global unsigned int *rnd_buffer, __global float *result, uint itt)
    {
    int id = get_global_id(0);
    if (id < datasize)
        {
        for (uint i=0; i<itt; i++)
            {
             result[id + i*datasize] = wang_rnd(rnd_buffer, id);
            }
        }
}
