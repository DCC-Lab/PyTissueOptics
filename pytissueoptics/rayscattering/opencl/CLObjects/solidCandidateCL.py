from pytissueoptics.rayscattering.opencl.CLObjects.CLObject import *


class SolidCandidateCL(CLObject):
    STRUCT_NAME = "SolidCandidate"
    STRUCT_DTYPE = np.dtype(
            [("distance", cl.cltypes.float),
             ("solidID", cl.cltypes.uint)])

    def __init__(self, nWorkUnits: int, nSolids: int):
        self._size = nWorkUnits * nSolids
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(self._size, 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        buffer["distance"] = -1
        buffer["solidID"] = 0
        return buffer
