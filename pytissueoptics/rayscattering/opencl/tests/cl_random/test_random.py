import os
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

context = cl.create_some_context()
mainQueue = cl.CommandQueue(context)
device = context.devices[0]
program = None
N = 1000000
kernelPath = os.path.dirname(os.path.abspath(__file__)) + "{}rand.c".format(os.sep)

HOST_randomSeed = np.random.randint(low=0, high=2 ** 32 - 1, size=N, dtype=cl.cltypes.uint)
DEVICE_randomSeed = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=HOST_randomSeed)
HOST_randomFloat = np.empty(N, dtype=cl.cltypes.float)
DEVICE_randomFloat = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=HOST_randomFloat)

with open(kernelPath, "r") as kernelFile:
    kernelSource = kernelFile.read()
    program = cl.Program(context, kernelSource).build()

program.fillRandomFloatBuffer(mainQueue, (N,), None, DEVICE_randomSeed, DEVICE_randomFloat)
cl.enqueue_copy(mainQueue, HOST_randomFloat, DEVICE_randomFloat)

plt.imshow(HOST_randomFloat.reshape((1000, 1000)))
plt.show()
