import pyopencl as cl
import numpy as np
from numpy.random import Generator


kernelSource = """

__kernel void testKernel(uint size, __global float *result, __global float *randomNum, uint randomIndex)
{
    int gid = get_global_id(0);
    if (gid < size)
        result[gid] = randomNum[gid + randomIndex];
    randomIndex++;
}

__kernel void testKernel2(__global float *result, __global float *randomNum, uint size, uint itt){
    
    int gid = get_global_id(0);
    
    testKernel(size, result, randomNum, randomIndex);
    
}
"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)
device = context.devices[0]
program = cl.Program(context, kernelSource).build()

N = 4

HOST_result = np.ones(N, dtype=cl.cltypes.float)
DEVICE_result = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_result)

rng = np.random.default_rng()
HOST_rand = rng.random(size=N*2, dtype=cl.cltypes.float)
print(HOST_rand)
DEVICE_rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_rand)

program.testKernel2(queue, HOST_result.shape, None, DEVICE_result, DEVICE_rand, np.int32(4), np.int32(0))
queue.finish()
cl.enqueue_copy(queue, dest=HOST_result, src=DEVICE_result)


print(HOST_result)


"""
Using cl.Buffer() with the COPY_HOST_PTR flag, the host buffer is copied to the device buffer and it allocates memory
the size of the buffer where HOST_PTR points to.
"""