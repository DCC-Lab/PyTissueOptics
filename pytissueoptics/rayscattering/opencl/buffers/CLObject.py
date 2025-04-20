import numpy as np

try:
    import pyopencl as cl
except ImportError:

    class DummyType:
        def __getattr__(self, item):
            return None

    cl = DummyType()
    cl.cltypes = DummyType()


class CLObject:
    STRUCT_NAME = None
    STRUCT_DTYPE = None

    def __init__(self, skipDeclaration: bool = False, buildOnce: bool = False):
        self._declaration = None
        self._dtype = None
        self._skipDeclaration = skipDeclaration
        self._buildOnce = buildOnce

        self._HOST_buffer = None
        self._DEVICE_buffer = None

    def build(self, device: cl.Device, context):
        if self.deviceBuffer is not None:
            if self._buildOnce:
                return
        self.make(device)
        self._DEVICE_buffer = cl.Buffer(
            context, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, hostbuf=self.hostBuffer
        )

    def make(self, device):
        if self.STRUCT_DTYPE:
            cl_struct, self._declaration = cl.tools.match_dtype_to_c_struct(device, self.STRUCT_NAME, self.STRUCT_DTYPE)
            self._dtype = cl.tools.get_or_register_dtype(self.STRUCT_NAME, cl_struct)

    def reset(self):
        self.hostBuffer = self._getInitialHostBuffer()

    def _getInitialHostBuffer(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.STRUCT_NAME

    @property
    def declaration(self) -> str:
        if not self._declaration or self._skipDeclaration:
            return ""
        return self._declaration

    @property
    def dtype(self) -> ...:
        assert self._dtype is not None
        return self._dtype

    @property
    def hostBuffer(self):
        if self._HOST_buffer is None:
            self._HOST_buffer = self._getInitialHostBuffer()
        return self._HOST_buffer

    @hostBuffer.setter
    def hostBuffer(self, value):
        if isinstance(value, np.ndarray):
            self._HOST_buffer = value
        else:
            raise TypeError("hostBuffer must be a numpy.ndarray")

    @property
    def deviceBuffer(self):
        return self._DEVICE_buffer

    @property
    def length(self) -> int:
        return len(self._HOST_buffer)

    @property
    def nBytes(self) -> int:
        if self.STRUCT_DTYPE is None:
            return self._HOST_buffer.nbytes
        return self.getItemSize() * self.length

    @classmethod
    def getItemSize(cls) -> int:
        """Returns the size of a single item in bytes aligned with the next power of 2."""
        if cls.STRUCT_DTYPE is None:
            raise NotImplementedError()

        itemSize = cls.STRUCT_DTYPE.itemsize
        alignedItemSize = 2 ** (itemSize - 1).bit_length()
        return alignedItemSize


class EmptyBuffer(CLObject):
    def __init__(self, N: int):
        self._N = N
        super().__init__()

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.empty(self._N, dtype=np.float32)


class RandomBuffer(CLObject):
    def __init__(self, N: int):
        self._N = N
        super().__init__()

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.random.rand(self._N).astype(np.float32)


class BufferOf(CLObject):
    def __init__(self, array: np.ndarray):
        self._array = array
        super().__init__()

    def _getInitialHostBuffer(self) -> np.ndarray:
        return self._array
