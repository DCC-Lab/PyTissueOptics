import numpy as np
import pyopencl as cl
import pyopencl.tools


def makePhotonType(device: cl.Device):
    photonStruct = np.dtype(
        [("position", cl.cltypes.float4),
         ("direction", cl.cltypes.float4),
         ("er", cl.cltypes.float4),
         ("weight", cl.cltypes.float),
         ("material_id", cl.cltypes.uint)])
    name = "photonStruct"
    photonStruct, c_decl_photon = cl.tools.match_dtype_to_c_struct(device, name, photonStruct)
    photon_dtype = cl.tools.get_or_register_dtype(name, photonStruct)
    return photon_dtype, c_decl_photon


def makeMaterialType(device: cl.Device):
    materialStruct = np.dtype(
        [("mu_s", cl.cltypes.float),
         ("mu_a", cl.cltypes.float),
         ("mu_t", cl.cltypes.float),
         ("g", cl.cltypes.float),
         ("n", cl.cltypes.float),
         ("albedo", cl.cltypes.float),
         ("material_id", cl.cltypes.uint)])
    name = "materialStruct"
    materialStruct, c_decl_mat = cl.tools.match_dtype_to_c_struct(device, name, materialStruct)
    material_dtype = cl.tools.get_or_register_dtype(name, materialStruct)
    return material_dtype, c_decl_mat


def makeLoggerType(device: cl.Device):
    loggerStruct = np.dtype(
        [("delta_weight", cl.cltypes.float),
         ("x", cl.cltypes.float),
         ("y", cl.cltypes.float),
         ("z", cl.cltypes.float)])
    name = "loggerStruct"
    loggerStruct, c_decl_logger = cl.tools.match_dtype_to_c_struct(device, name, loggerStruct)
    logger_dtype = cl.tools.get_or_register_dtype(name, loggerStruct)
    return logger_dtype, c_decl_logger
