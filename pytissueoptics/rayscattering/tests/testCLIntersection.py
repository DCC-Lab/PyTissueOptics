import os
import traceback
import unittest

import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics import *
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE, OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.CLPhotons import CLScene

if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import CLObject


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLIntersection(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "intersection.c")
        self.program = CLProgram(sourcePath)

    def testLaunchKernel(self):
        N = 10
        _scene = self._getTestScene()
        clScene = CLScene(_scene, N)

        rays = RayCL(origins=np.full((N, 3), [0, 0, -2]),
                     directions=np.full((N, 3), [0, 0, 1]),
                     lengths=np.full(N, 2.5))
        intersections = IntersectionCL(N)

        try:
            self.program.launchKernel("findIntersections", N=N, arguments=[clScene.nSolids, rays, clScene.solids,
                                                                           clScene.bboxIntersections, intersections])
        except Exception as e:
            traceback.print_exc(0)

        self.program.getData(clScene.bboxIntersections)
        self.program.getData(intersections)

        print("BBox Intersections: ", clScene.bboxIntersections.hostBuffer)
        print("Intersections: ", intersections.hostBuffer)

    def _getTestScene(self):
        material1 = ScatteringMaterial(0.1, 0.8, 0.8, 1.4)
        material2 = ScatteringMaterial(2, 0.8, 0.8, 1.2)
        material3 = ScatteringMaterial(0.5, 0.8, 0.8, 1.3)

        layer1 = Cuboid(a=10, b=10, c=2, position=Vector(0, 0, 0), material=material1, label="Layer 1")
        layer2 = Cuboid(a=10, b=10, c=2, position=Vector(0, 0, 0), material=material2, label="Layer 2")
        tissue = layer1.stack(layer2, "back")
        solid2 = Cuboid(2, 2, 2, position=Vector(10, 0, 0), material=material3)
        scene = RayScatteringScene([tissue, solid2], worldMaterial=ScatteringMaterial())
        return scene


class RayCL(CLObject):
    STRUCT_NAME = "Ray"

    def __init__(self, origins: np.ndarray, directions: np.ndarray, lengths: np.ndarray):
        self._origins = origins
        self._directions = directions
        self._lengths = lengths
        self._N = origins.shape[0]

        struct = np.dtype([("origin", cl.cltypes.float4),
                           ("direction", cl.cltypes.float4),
                           ("length", cl.cltypes.float)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, skipDeclaration=True)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._origins
        buffer[:, 4:7] = self._directions
        buffer[:, 8] = self._lengths
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
