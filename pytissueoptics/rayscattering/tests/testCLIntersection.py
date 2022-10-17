import os
import traceback
import unittest

import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE, OPENCL_SOURCE_DIR

if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import CLObject, BBoxIntersectionCL


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLIntersection(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "intersection.c")
        self.program = CLProgram(sourcePath)

    def testLaunchKernel(self):
        nSolids = 8
        N = 10
        rays = RayCL(np.zeros((N, 3)), np.ones((N, 3)), np.full(N, 2.5))
        intersections = IntersectionCL(N)
        bboxIntersections = BBoxIntersectionCL(N, nSolids)
        workUnits = np.uint32(N)

        try:
            self.program.launchKernel("findIntersections", N=N, arguments=[rays, bboxIntersections, workUnits,
                                                                           intersections])
        except Exception as e:
            traceback.print_exc(0)

        self.program.getData(intersections)
        self.program.getData(bboxIntersections)

        print(intersections.hostBuffer)
        print(bboxIntersections.hostBuffer)


class RayCL(CLObject):
    STRUCT_NAME = "Ray"

    def __init__(self, origins: np.ndarray, directions: np.ndarray, distances: np.ndarray):
        self._origins = origins
        self._directions = directions
        self._distances = distances
        self._N = origins.shape[0]

        struct = np.dtype([("origin", cl.cltypes.float4),
                           ("direction", cl.cltypes.float4),
                           ("distance", cl.cltypes.float)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, skipDeclaration=True)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._origins
        buffer[:, 4:7] = self._directions
        buffer[:, 8] = self._distances
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        return buffer


class IntersectionCL(CLObject):
    STRUCT_NAME = "Intersection"

    def __init__(self, N: int):
        self._N = N
        struct = np.dtype([("status", cl.cltypes.uint),
                           ("distance", cl.cltypes.float)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, skipDeclaration=True)

    def _getHostBuffer(self) -> np.ndarray:
        return np.empty(self._N, dtype=self._dtype)


if __name__ == '__main__':
    unittest.main()
