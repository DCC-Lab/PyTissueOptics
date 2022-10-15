import os
from typing import List

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl.CLObjects import CLObject


class CLProgram:
    def __init__(self, sourcePath: str):
        self._sourcePath = sourcePath
        self._context = cl.create_some_context()
        self._mainQueue = cl.CommandQueue(self._context)
        self._device = self._context.devices[0]
        self._program = None

    def launchKernel(self, kernelName: str, N: int, arguments: list):
        self._build(objects=[x for x in arguments if isinstance(x, CLObject)])
        sizeOnDevice = sum([x.hostBuffer.nbytes for x in arguments if isinstance(x, CLObject)])
        arguments = [x.deviceBuffer if isinstance(x, CLObject) else x for x in arguments]

        kernel = getattr(self._program, kernelName)
        try:
            kernel(self._mainQueue, (N,), None, *arguments)
        except cl.MemoryError:
            raise MemoryError(f"Cannot allocate {sizeOnDevice//10**6} MB of memory on the device; "
                              f"the buffers are too large.")
        self._mainQueue.finish()

    def _build(self, objects: List[CLObject]):
        for _object in objects:
            _object.build(self._device, self._context)

        typeDeclarations = ''.join([_object.declaration for _object in objects])
        sourceCode = typeDeclarations + self._makeSource(self._sourcePath)

        self._program = cl.Program(self._context, sourceCode).build()

    def getData(self, _object: CLObject):
        cl.enqueue_copy(self._mainQueue, dest=_object.hostBuffer, src=_object.deviceBuffer)
        return rfn.structured_to_unstructured(_object.hostBuffer)

    def showDeviceInfo(self):
        devices = self._context.devices  # type: List[cl.Device]
        print("Available devices:")
        for i, device in enumerate(devices):
            print(f"... Device {i}: {device.name} ({device.global_mem_size // 10**6} MB "
                  f"| {device.max_clock_frequency} MHz)")

    @staticmethod
    def _makeSource(sourcePath) -> str:
        includeStatement = ''

        libFiles = []
        with open(sourcePath, 'r') as f:
            line = f.readline()
            while line.startswith("#include"):
                includeStatement += line
                libFiles.append(line.split()[-1].strip('"'))
                line = f.readline()

        includeDir = os.path.dirname(sourcePath)
        sourceCode = ""
        for libFile in libFiles:
            sourceCode += open(os.path.join(includeDir, libFile)).read()

        mainCode = open(sourcePath).read()[len(includeStatement):]
        sourceCode += mainCode
        return sourceCode
