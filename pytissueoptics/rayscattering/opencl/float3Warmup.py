from pytissueoptics import *
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array
import pyopencl.tools
import numpy as np

context = cl.create_some_context()
queue = cl.CommandQueue(context)
kernelSource = open("./float3Warmup.c").read()
device = context.devices[0]
program = cl.Program(context,  kernelSource).build()

N = 10000

HOST_float3 = np.array([[0,0,1]]*N, dtype=np.float32)
TARGET_float3 = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_float3)
cl.enqueue_copy(queue, dest=TARGET_float3, src=HOST_float3)
program.vectorDouble(queue=queue, global_size=N, local_size=None, args=[TARGET_float3])
cl.enqueue_copy(queue, dest=HOST_float3, src=TARGET_float3)
queue.finish()

print(HOST_float3)
