from pytissueoptics.rayscattering.opencl.buffers.CLObject import *


class SeedCL(CLObject):
    def __init__(self, size: int, seed: int = None):
        self._size = size
        if seed is not None:
            np.random.seed(seed)

        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.random.randint(low=0, high=2 ** 32 - 1, size=self._size, dtype=cl.cltypes.uint)
