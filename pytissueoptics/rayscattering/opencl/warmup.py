from pytissueoptics import *
import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import numpy as np

kernelSource = """

__kernel void moveByOne(__global photonStruct *a, const int n)
{
    int ix = get_global_id(0);
    a[ix].position = a[ix].position + n * a[ix].direction;
}
"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)

def makePhotonType(device):
    myStruct = np.dtype(
        [("position", (np.float, 3)), ("direction", (np.float, 3)), ("er", (np.float, 3)), ("weight", np.float),
         ("materialId", np.uint8), ("fresnelIntersectId", np.uint8), ("loggerId", np.uint8)])
    name = "photonStruct"
    dtype, c_decl = cl.tools.match_dtype_to_c_struct(device, name, myStruct)
    dtype = cl.tools.get_or_register_dtype(name, myStruct)
    return dtype, c_decl


photon_dtype, photon_c_decl = makePhotonType(context.devices[0])
print(photon_c_decl)

program = cl.Program(context, photon_c_decl+kernelSource).build()
N = 10000
HOST_photons = np.empty(N, dtype=photon_dtype)
HOST_photons["position"].fill(np.array([0.0, 0.0, 0.0]))
HOST_photons["direction"].fill(np.array([0.0, 0.0, 1.0]))
HOST_photons["er"].fill(np.array([0.0, 1.0, 0.0]))
HOST_photons["weight"].fill(1.0)
HOST_photons["materialId"].fill(0)
HOST_photons["fresnelIntersectId"].fill(0)
HOST_photons["loggerId"].fill(0)

print(HOST_photons)

TARGET_photons = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_photons)
cl.enqueue_copy(queue, dest=TARGET_photons, src=HOST_photons)
program.moveByOne(queue, (N,), None, TARGET_photons, 1)
cl.enqueue_copy(queue, dest=HOST_photons, src=TARGET_photons)
queue.finish()

print(HOST_photons)
