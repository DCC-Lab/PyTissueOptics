import opencl
import numpy
import time
import sys
from tools.tools import *


@timeit
def run_ocl_kernel(queue, kernel, global_size, input_tuples, output_tuples, local_size=(32,), output_data=False):
    """
    A function that launches the memory copy to the device, executes the kernel
      and recopies the array to the host computer memory
    """
    # copying data onto the device
    for (array, buffer) in input_tuples:
        opencl.enqueue_copy(queue, src=array, dest=buffer)

    # running program on the device
    kernel_arguments = [buffer for (_, buffer) in input_tuples]
    kernel_arguments += [buffer for (_, buffer) in output_tuples]

    kernel(queue, global_size, local_size,
           *kernel_arguments)

    # copying data off the device
    if output_data:
        for (arr, buffer) in output_tuples:
            opencl.enqueue_copy(queue, src=buffer, dest=arr)

    # waiting for everything to finish
    queue.finish()

# VERIFY WHAT INTERFACE YOU CAN USE
ocl_platforms = (platform.name
                 for platform in opencl.get_platforms())
print("\n".join(ocl_platforms))

# GET YOUR INTERFACE/DEVICE
nvidia_platform = [platform
                   for platform in opencl.get_platforms()
                   if platform.name == "NVIDIA CUDA"][0]
nvidia_devices = nvidia_platform.get_devices()

nvidia_context = opencl.Context(devices=nvidia_devices)

program_source = """
      kernel void sum(global float *a, 
                      global float *b,
                      global float *c){
        int gid = get_global_id(0);
        c[gid] = a[gid] + b[gid];
      }
    """
nvidia_program_source = opencl.Program(nvidia_context, program_source)
nvidia_program = nvidia_program_source.build()

program_kernel_names = nvidia_program.get_info(opencl.program_info.KERNEL_NAMES)
print("Kernel Names: {}".format(program_kernel_names))

# Synthetic data setup
N = int(2**24)
a = numpy.random.rand(N).astype(numpy.float32)
b = numpy.random.rand(N).astype(numpy.float32)
c = numpy.empty_like(a)

# Device Memory setup
a_nvidia_buffer = opencl.Buffer(nvidia_context,
                                flags=opencl.mem_flags.READ_ONLY,
                                size=a.nbytes)
b_nvidia_buffer = opencl.Buffer(nvidia_context,
                                flags=opencl.mem_flags.READ_ONLY,
                                size=b.nbytes)
c_nvidia_buffer = opencl.Buffer(nvidia_context,
                                flags=opencl.mem_flags.WRITE_ONLY,
                                size=c.nbytes)

nvidia_queue = opencl.CommandQueue(nvidia_context)

input_tuples = ((a, a_nvidia_buffer), (b, b_nvidia_buffer), )
output_tuples = ((c, c_nvidia_buffer),)
run_ocl_kernel(nvidia_queue, nvidia_program.sum, (N,), input_tuples, output_tuples)

def create_input_memory(context, input_arrays):
    return [(array, opencl.Buffer(context,
                                  flags=opencl.mem_flags.READ_ONLY,
                                  size=array.nbytes))
        for array in input_arrays]

def create_output_memory(context, output_arrays):
    return [(array, opencl.Buffer(context,
                                  flags=opencl.mem_flags.WRITE_ONLY,
                                  size=array.nbytes))
        for array in output_arrays]



def cleanup_memories(tuples):
    for (_, buffer) in tuples:
        buffer.release()