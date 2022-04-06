import pyopencl as cl
import pyopencl.cltypes
import pyopencl.tools
import numpy as np


kernelSource = """
// GENERAL FUNCTIONS

__kernel void random(uint2 seed, __global float *buffer) {
    int idx = get_global_id(0);
    uint seed = randoms.x + idx;
    uint t = seed ^ (seed << 11);  
    uint result = randoms.y ^ (randoms.y >> 19) ^ (t ^ (t >> 8));
    buffer[idx] = result

// PHOTON PHYSICS
__kernel void decreaseWeightBy(__global photonStruct *photon, float delta_weight)
{
    int gid = get_global_id(0);
        photon[gid].weight -= delta_weight;
}


__kernel void interact(__global photonStruct *photon, __constant materialStruct *mat)
{
    int gid = get_global_id(0);
    float delta = photon[gid].weight * mat[photon[gid].material_id].albedo;
    decreaseWeightBy(photon, delta);
}


__kernel void moveBy(__global photonStruct *photon, float distance)
{
    int gid = get_global_id(0);
        photon[gid].position += distance * photon[gid].direction;
}

__kernel void roulette(__global photonStruct *photon, float *survival_probability, float *randomBuffer){
    int gid = get_global_id(0);
    random(uint2(1, 2), randomBuffer);
    float chance = 0.1;
    if (photon[gid].weight >= 0.0001){return;}
    else if (randomBuffer[gid] < chance){
        photon[gid].weight /= chance;
    }
    else{
        photon[gid].weight = 0; 
    }
    
}


__kernel void propagate(__global photonStruct *photons)
{
    int gid = get_global_id(0);
    while(photons[gid].weight > 0)
    {
        moveBy(photons, 1.0f);
        decreaseWeightBy(photons, 0.3f);
    }
}

// MATERIAL KERNELS
__kernel void getScatteringDistances(__constant materialStruct *material, int amount)
{
    int gid = get_global_id(0);
    
}


"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)
device = context.devices[0]


def makePhotonType():
    photonStruct = np.dtype(
        [("position", cl.cltypes.float4),
         ("direction", cl.cltypes.float4),
         ("weight", cl.cltypes.float),
         ("material_id", cl.cltypes.uint)])
    name = "photonStruct"
    photonStruct, c_decl_photon = cl.tools.match_dtype_to_c_struct(device, name, photonStruct)
    photon_dtype = cl.tools.get_or_register_dtype(name, photonStruct)
    return photon_dtype, c_decl_photon


def makeMaterialType():
    materialStruct = np.dtype(
        [("mu_s", cl.cltypes.float),
         ("mu_a", cl.cltypes.float),
         ("mu_t", cl.cltypes.float),
         ("g", cl.cltypes.float),
         ("n", cl.cltypes.float),
         ("albedo", cl.cltypes.float),
         ("material_id", cl.cltypes.uint)])
    name = "materialStruct"
    myPhotonStruct, c_decl_mat = cl.tools.match_dtype_to_c_struct(device, name, materialStruct)
    material_dtype = cl.tools.get_or_register_dtype(name, myPhotonStruct)
    return material_dtype, c_decl_mat


photon_dtype, c_decl_photon = makePhotonType()
material_dtype, c_decl_mat = makeMaterialType()
program = cl.Program(context, c_decl_photon + c_decl_mat + kernelSource).build()

# PHOTONS
N = 10000
HOST_photons = np.empty(N, dtype=photon_dtype)
HOST_photons["position"] = np.array([cl.cltypes.make_float4(1, 0, 0, 0)]*N, dtype=cl.cltypes.float4)
HOST_photons["direction"] = np.array([cl.cltypes.make_float4(0, 0, 1, 0)]*N, dtype=cl.cltypes.float4)
HOST_photons["weight"] = np.ones(N, dtype=cl.cltypes.float)
HOST_photons["material_id"] = np.zeros(N, dtype=cl.cltypes.float)
TARGET_photons = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_photons)
cl.enqueue_copy(queue, dest=TARGET_photons, src=HOST_photons)

# MATERIALS
HOST_material = np.empty(1, dtype=material_dtype)
HOST_material["mu_s"] = 0.5
HOST_material["mu_a"] = 0.5
HOST_material["mu_t"] = HOST_material["mu_s"] + HOST_material["mu_a"]
HOST_material["g"] = 0.5
HOST_material["n"] = 1.5
HOST_material["albedo"] = HOST_material["mu_a"] / HOST_material["mu_t"]
HOST_material["material_id"] = 0
TARGET_material = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_material)
cl.enqueue_copy(queue, dest=TARGET_material, src=HOST_material)

# RANDOM FLOATS [0, 1]
HOST_rand = np.random.random(size=1000000)
TARGET_rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_rand)
cl.enqueue_copy(queue, dest=TARGET_rand, src=HOST_rand)

# program.decreaseWeightBy(queue, (N,), None, TARGET_photons, np.float32(1.0))
# program.moveBy(queue, (N,), None, TARGET_photons, np.float32(1.0))
# while not all(TARGET_photons["weight"] <= 0):
for i in range(10):
    program.moveBy(queue, (N,), None, TARGET_photons, np.float32(1.0))
    program.interact(queue, (N,), None, TARGET_photons, TARGET_material)
    #program.getScatteringDistances(queue, (1,), None, TARGET_photons, TARGET_material, TARGET_rand)
    #program.propagate(queue, (N,), None, TARGET_photons)
cl.enqueue_copy(queue, dest=HOST_photons, src=TARGET_photons)
queue.finish()

print(HOST_photons)
