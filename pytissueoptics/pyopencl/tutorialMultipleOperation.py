import pyopencl as pycl
import numpy
import time
import sys
from tools.tools import *


def create_read_write_memory(context, arrays):
    return [(array, pycl.Buffer(context,
                                flags=pycl.mem_flags.WRITE_ONLY,
                                size=array.nbytes))
            for array in arrays]

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
N = int(2 ** 22)
a = numpy.ones(N).astype(numpy.float32)
b = numpy.ones(N).astype(numpy.float32)
c = numpy.empty_like(a)

# Device Memory setup
memTuples = create_read_write_memory(context, [a, b, c])
queue = pycl.CommandQueue(context)
for (array, buffer) in memTuples:
    pycl.enqueue_copy(queue, src=array, dest=buffer)

a_buf, b_buf, c_buf = [memTuple[1] for memTuple in memTuples]

program.sum(queue, (N,), (32,), *[a_buf, b_buf, c_buf])
program.sum(queue, (N,), (32,), *[c_buf, b_buf, c_buf])
program.sum(queue, (N,), (32,), *[c_buf, b_buf, c_buf])
program.sum(queue, (N,), (32,), *[c_buf, b_buf, c_buf])
program.sum(queue, (N,), (32,), *[c_buf, b_buf, c_buf])

for (arr, buffer) in memTuples:
    pycl.enqueue_copy(queue, src=buffer, dest=arr)

queue.finish()

print(a)
print(b)
print(c)
