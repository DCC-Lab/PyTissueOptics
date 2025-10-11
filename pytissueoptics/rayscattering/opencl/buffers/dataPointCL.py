import numpy as np

from .CLObject import CLObject, cl


class DataPointCL(CLObject):
    STRUCT_NAME = "DataPoint"

    STRUCT_DTYPE = np.dtype(
        [
            ("delta_weight", cl.cltypes.float),
            ("x", cl.cltypes.float),
            ("y", cl.cltypes.float),
            ("z", cl.cltypes.float),
            ("photonID", cl.cltypes.uint),
            ("solidID", cl.cltypes.int),
            ("surfaceID", cl.cltypes.int),
        ]
    )

    def __init__(self, size: int):
        self._size = size
        super().__init__()

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.zeros(self._size, dtype=self._dtype)
