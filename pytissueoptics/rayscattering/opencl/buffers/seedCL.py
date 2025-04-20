import numpy as np

from .CLObject import CLObject, cl


class SeedCL(CLObject):
    def __init__(self, size: int):
        self._size = size
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.random.randint(low=0, high=2 ** 32 - 1, size=self._size, dtype=cl.cltypes.uint)
