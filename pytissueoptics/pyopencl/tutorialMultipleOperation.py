import pyopencl as pycl
import numpy as np
import time
import sys
from tools.tools import *


def create_read_write_memory(context, arrays):
    return [(array, pycl.Buffer(context,
                                flags=pycl.mem_flags.WRITE_ONLY,
                                size=array.nbytes))
            for array in arrays]


@timeit
def executeWithoutBufferMovement(itterations):
    for i in range(itterations):
        program.sum(queue, (N,), (32,), *[c_buf, b_buf, c_buf])


@timeit
def executeNonParrallel(itterations, a, b):
    for i in range(itterations):
        b = np.add(a, b)

@timeit
def executeWithBufferMovement(itterations):
    for i in range(itterations):
        memTuples = create_read_write_memory(context, [a, b, c])
        for (array, buffer) in memTuples:
            pycl.enqueue_copy(queue, src=array, dest=buffer)
        a_buf, b_buf, c_buf = [memTuple[1] for memTuple in memTuples]

        program.sum(queue, (N,), (32,), *[c_buf, b_buf, c_buf])
        for (arr, buffer) in memTuples:
            pycl.enqueue_copy(queue, src=buffer, dest=arr)

        queue.finish()

N = int(2 ** 25)
print("testing 100 successive (+) operations on (1, 6.0E+6) scalar arrays")

nvidia_platform = [platform for platform in pycl.get_platforms()if platform.name == "NVIDIA CUDA"][0]
nvidia_devices = nvidia_platform.get_devices()
context = pycl.Context(devices=nvidia_devices)

program_source = """
      kernel void sum(global float *a, 
                      global float *b,
                      global float *c){
        int gid = get_global_id(0);
        c[gid] = a[gid] + b[gid];
      }

        kernel void multiply(global float *a, 
                      global float *b,
                      global float *c){
        int gid = get_global_id(0);
        c[gid] = a[gid] * b[gid];
      }
    """
program_source = pycl.Program(context, program_source)
program = program_source.build()

# Synthetic data setup

a = np.ones(N).astype(np.float32)
b = np.ones(N).astype(np.float32)
c = np.empty_like(a)

# Device Memory setup
memTuples = create_read_write_memory(context, [a, b, c])
queue = pycl.CommandQueue(context)
for (array, buffer) in memTuples:
    pycl.enqueue_copy(queue, src=array, dest=buffer)
a_buf, b_buf, c_buf = [memTuple[1] for memTuple in memTuples]
print(a_buf, b_buf, c_buf)

program.sum(queue, (N,), (32,), *[a_buf, b_buf, c_buf])

executeWithoutBufferMovement(100)

for (arr, buffer) in memTuples:
    pycl.enqueue_copy(queue, src=buffer, dest=arr)

queue.finish()


a = np.ones(N).astype(np.float32)
b = np.ones(N).astype(np.float32)
c = np.empty_like(a)
executeWithBufferMovement(100)


a = np.ones(N).astype(np.float32)
b = np.ones(N).astype(np.float32)
executeNonParrallel(100, a, b)
