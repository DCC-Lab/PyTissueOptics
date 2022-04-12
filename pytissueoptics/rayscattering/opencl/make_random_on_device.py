import pyopencl as cl
import numpy as np
from numpy.random import Generator

kernelSource = """

uint wang_hash(uint seed)
    {
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
    }

void wang_rnd_0(__global unsigned int * rnd_buffer, int id)                
    {
     uint rnd_int = wang_hash(id);
     rnd_buffer[id] = rnd_int;
    }
 
 __kernel void rnd_init(__global unsigned int *rnd_buffer)
    {
       int id=get_global_id(0);
       wang_rnd_0(rnd_buffer, id);  // each (id) thread has its own random seed now           
    }
 
float wang_rnd(__global unsigned int * rnd_buffer, int id)                
    {
     uint maxint = 0;
     maxint--;
     uint rnd_int = wang_hash(rnd_buffer[id]);
     rnd_buffer[id] = rnd_int;
     return ((float)rnd_int)/(float)maxint;
    }

 __kernel void rnd_1(__global unsigned int * rnd_buffer)
    {
    int id=get_global_id(0);
    
    // can use this to populate a buffer with random numbers 
    // concurrently on all cores of a gpu
    float thread_private_random_number=wang_rnd(rnd_buffer,id);
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
"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)
device = context.devices[0]
program = cl.Program(context, kernelSource).build()

N = 2
itt = 10

HOST_result = np.zeros(N*itt, dtype=cl.cltypes.float)
DEVICE_result = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_result)

HOST_rand = np.zeros(N, dtype=cl.cltypes.uint)
DEVICE_rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_rand)

# rng = np.random.default_rng()
# HOST_rand = np.random.randint(size=N, low=0, high=2**32-1, dtype=cl.cltypes.uint)
# DEVICE_rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_rand)

program.rnd_init(queue, (N,), None, DEVICE_rand)
program.testKernel(queue, HOST_rand.shape, None, np.uint32(N), DEVICE_rand, DEVICE_result, np.uint32(itt))
queue.finish()
cl.enqueue_copy(queue, dest=HOST_result, src=DEVICE_result)
cl.enqueue_copy(queue, dest=HOST_rand, src=DEVICE_rand)

print(HOST_rand)
print(HOST_result)

"""
Using cl.Buffer() with the COPY_HOST_PTR flag, the host buffer is copied to the device buffer and it allocates memory
the size of the buffer where HOST_PTR points to.
"""