import numpy as np
try:
    import pyopencl as cl
    import pyopencl.tools
except ImportError:
    pass


class CLType:
    def __init__(self, name: str, struct: np.dtype, device: 'cl.Device'):
        self._name = name

        cl_struct, self._declaration = cl.tools.match_dtype_to_c_struct(device, self._name, struct)
        self._dtype = cl.tools.get_or_register_dtype(self._name, cl_struct)

    @property
    def name(self) -> str:
        return self._name

    @property
    def declaration(self) -> str:
        return self._declaration

    @property
    def dtype(self) -> ...:
        return self._dtype


class PhotonCLType(CLType):
    def __init__(self, device: 'cl.Device'):
        photonStruct = np.dtype(
            [("position", cl.cltypes.float4),
             ("direction", cl.cltypes.float4),
             ("er", cl.cltypes.float4),
             ("weight", cl.cltypes.float),
             ("material_id", cl.cltypes.uint)])
        super().__init__(name="photonStruct", struct=photonStruct, device=device)


def makePhotonType(device: 'cl.Device'):
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


def makeMaterialType(device: 'cl.Device'):
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


def makeLoggerType(device: 'cl.Device'):
    loggerStruct = np.dtype(
        [("delta_weight", cl.cltypes.float),
         ("x", cl.cltypes.float),
         ("y", cl.cltypes.float),
         ("z", cl.cltypes.float)])
    name = "loggerStruct"
    loggerStruct, c_decl_logger = cl.tools.match_dtype_to_c_struct(device, name, loggerStruct)
    logger_dtype = cl.tools.get_or_register_dtype(name, loggerStruct)
    return logger_dtype, c_decl_logger
