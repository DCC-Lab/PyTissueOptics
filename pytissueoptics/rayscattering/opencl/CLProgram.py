import os
import time
from typing import List

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    pass

from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl import CONFIG
from pytissueoptics.rayscattering.opencl.buffers import CLObject


class CLProgram:
    def __init__(self, sourcePath: str):
        self._sourcePath = sourcePath
        self._context = CONFIG.clContext
        self._device = CONFIG.device

        self._mainQueue = cl.CommandQueue(self._context)
        self._program = None
        self._include = ''
        self._mocks = []

    def launchKernel(self, kernelName: str, N: int, arguments: list, verbose: bool = False):
        t0 = time.time()
        CLObjects = [arg for arg in arguments if isinstance(arg, CLObject)]
        self._build(CLObjects)
        if verbose:
            for _object in CLObjects:
                print(f" ... {_object.name} ({_object.nBytes / 1024**2:.3f} MB)")
        sizeOnDevice = sum([x.nBytes for x in CLObjects])
        buffers = self._extractBuffers(arguments)
        t1 = time.time()
        if verbose:
            print(f" ... {t1 - t0:.3f} s. [Build]")

        kernel = getattr(self._program, kernelName)
        try:
            kernel(self._mainQueue, (N,), None, *buffers)
        except cl.MemoryError:
            raise MemoryError(f"Cannot allocate {sizeOnDevice//1024**2} MB on the device;"
                              f"the buffers are too large.")
        self._mainQueue.finish()
        t2 = time.time()

        if verbose:
            print(f" ... {t2 - t1:.3f} s. [Kernel execution]")

    def _build(self, objects: List[CLObject]):
        for _object in objects:
            _object.build(self._device, self._context)

        typeDeclarations = ''.join([_object.declaration for _object in objects])
        sourceCode = self._include + typeDeclarations + self._makeSource(self._sourcePath)

        for code, mock in self._mocks:
            sourceCode = sourceCode.replace(code, mock)

        self._program = cl.Program(self._context, sourceCode).build()

    def getData(self, _object: CLObject, dtype: np.dtype = np.float32):
        cl.enqueue_copy(self._mainQueue, dest=_object.hostBuffer, src=_object.deviceBuffer)
        if _object.STRUCT_DTYPE is not None:
            return rfn.structured_to_unstructured(_object.hostBuffer, dtype=dtype)
        else:
            return _object.hostBuffer

    def include(self, code: str):
        self._include += code

    @staticmethod
    def _makeSource(sourcePath) -> str:
        includeDir = os.path.dirname(sourcePath)
        sourceCode = ''
        with open(sourcePath, 'r') as f:
            line = f.readline()
            while line.startswith("#include"):
                libFileName = line.split('"')[1]
                with open(os.path.join(includeDir, libFileName), 'r') as libFile:
                    sourceCode += libFile.read()
                line = f.readline()
            sourceCode += line
            sourceCode += f.read()

        return sourceCode

    @staticmethod
    def _extractBuffers(arguments):
        buffers = []
        for arg in arguments:
            if isinstance(arg, CLObject):
                buffers.append(arg.deviceBuffer)
            else:
                buffers.append(arg)
        return buffers

    @property
    def device(self):
        return self._device

    def mock(self, code: str, mock: str):
        """
        Used internally for testing purposes.
        Acts as a simple string replacement for the source code.
        """
        self._mocks.append((code, mock))
