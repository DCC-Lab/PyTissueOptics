import pyopencl as cl
import numpy as np
from numpy.random import Generator


kernelSource = """

__kernel void testKernel(__global float *result, __global float *randomNum)
{
    int gid = get_global_id(0);
    result[gid] = randomNum[gid];
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
HOST_rand = rng.random(size=N, dtype=cl.cltypes.float)
DEVICE_rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_rand)

program.testKernel(queue, HOST_result.shape, None, DEVICE_result, DEVICE_rand)
queue.finish()
cl.enqueue_copy(queue, dest=HOST_result, src=DEVICE_result)

print(HOST_rand)
print(HOST_result)


"""
Using cl.Buffer() with the COPY_HOST_PTR flag, the host buffer is copied to the device buffer and it allocates memory
the size of the buffer where HOST_PTR points to.
"""