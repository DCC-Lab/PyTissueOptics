from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl.buffers.CLObject import *


class PhotonCL(CLObject):
    STRUCT_NAME = "Photon"
    STRUCT_DTYPE = np.dtype(
            [("position", cl.cltypes.float3),
             ("direction", cl.cltypes.float3),
             ("er", cl.cltypes.float3),
             ("weight", cl.cltypes.float),
             ("materialID", cl.cltypes.uint),
             ("solidID", cl.cltypes.int)])

    def __init__(self, positions: np.ndarray, directions: np.ndarray,
                 materialID: int, solidID: int):
        self._positions = positions
        self._directions = directions
        self._N = positions.shape[0]
        self._materialID = materialID
        self._solidID = solidID

        super().__init__()

    def _getInitialHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._positions
        buffer[:, 4:7] = self._directions
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        buffer["weight"] = 1.0
        buffer["materialID"] = self._materialID
        buffer["solidID"] = self._solidID
        return buffer
