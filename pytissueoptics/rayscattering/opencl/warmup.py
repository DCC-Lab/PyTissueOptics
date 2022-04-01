from pytissueoptics import *
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array
import pyopencl.tools
import numpy as np

context = cl.create_some_context()
queue = cl.CommandQueue(context)
kernelSource = open("./warmup.c").read()
device = context.devices[0]


def makePhotonType(device):
    myStruct = np.dtype(
        [("position", cl.array.vec.float4), ("direction", cl.array.vec.float4)])
    name = "photonStruct"
    dtype, c_decl = cl.tools.match_dtype_to_c_struct(device, name, myStruct)
    dtype = cl.tools.get_or_register_dtype(name, myStruct)
    return dtype, c_decl

photon_dtype, photon_c_decl = makePhotonType(device)
program = cl.Program(context, photon_c_decl + kernelSource).build()
print(photon_c_decl)
N = 10000

HOST_photons = np.empty(N, dtype=photon_dtype)
HOST_photons["position"] = np.array([cl.array.vec.make_float4(1, 0, 0, 0)]*N, dtype=cl.array.vec.float4)
HOST_photons["direction"] = np.array([cl.array.vec.make_float4(0, 0, 1, 0)]*N, dtype=cl.array.vec.float4)


print(HOST_photons)

TARGET_photons = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_photons)
cl.enqueue_copy(queue, dest=TARGET_photons, src=HOST_photons)
program.moveByOne(queue, (N,), None, TARGET_photons)
cl.enqueue_copy(queue, dest=HOST_photons, src=TARGET_photons)
queue.finish()

print(HOST_photons)
