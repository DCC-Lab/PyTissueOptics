try:
    import pyopencl as cl
    import pyopencl.tools
except ImportError:
    pass

import numpy as np
from numpy.lib import recfunctions as rfn


class CLType:
    def __init__(self, name: str, struct: np.dtype):
        self._name = name
        self._struct = struct
        self._declaration = None
        self._dtype = None

        self._HOST_buffer = None
        self._DEVICE_buffer = None

    def build(self, device: 'cl.Device', context):
        cl_struct, self._declaration = cl.tools.match_dtype_to_c_struct(device, self._name, self._struct)
        self._dtype = cl.tools.get_or_register_dtype(self._name, cl_struct)

        self._makeHostBuffer()
        self._DEVICE_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self._HOST_buffer)

    def _makeHostBuffer(self):
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self._name

    @property
    def declaration(self) -> str:
        return self._declaration

    @property
    def dtype(self) -> ...:
        assert self._dtype is not None
        return self._dtype

    @property
    def deviceBuffer(self):
        return self._DEVICE_buffer


class PhotonCLType(CLType):
    def __init__(self, positions: np.ndarray, directions: np.ndarray):
        self._positions = positions
        self._directions = directions
        self._N = positions.shape[0]

        photonStruct = np.dtype(
            [("position", cl.cltypes.float4),
             ("direction", cl.cltypes.float4),
             ("er", cl.cltypes.float4),
             ("weight", cl.cltypes.float),
             ("material_id", cl.cltypes.uint)])
        super().__init__(name="photonStruct", struct=photonStruct)

    def _makeHostBuffer(self):
        photonsPrototype = np.zeros(self._N, dtype=self.dtype)
        photonsPrototype = rfn.structured_to_unstructured(photonsPrototype)
        photonsPrototype[:, 0:3] = self._positions[:, ::]
        photonsPrototype[:, 4:7] = self._directions[:, ::]
        photonsPrototype[:, 12] = 1.0
        photonsPrototype[:, 13] = 0
        self._HOST_buffer = rfn.unstructured_to_structured(photonsPrototype, self.dtype)


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
