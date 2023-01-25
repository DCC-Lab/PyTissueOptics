import os
import unittest
from dataclasses import dataclass

import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics import Vector, ScatteringMaterial
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from rayscattering.opencl.buffers import MaterialCL, SurfaceCL, SurfaceCLInfo, SeedCL


if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.buffers import CLObject


@dataclass
class FresnelResult:
    incidencePlane: Vector
    isReflected: bool
    angleDeflection: float
    nextMaterialID: int
    nextSolidID: int


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLFresnel(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "fresnel.c")
        self.program = CLProgram(sourcePath)
        self.program.include(self._getMissingSourceCode())

    def testWhenComputeFresnelIntersection(self):
        N = 1  # Kernel size errors when trying a vector buffer. Limiting to 1 for now, which is fine for testing.
        rayDirection = Vector(1, 0, 0)

        materials = MaterialCL([ScatteringMaterial(0.8, 0.2, 0.9, 1.4),
                                ScatteringMaterial(0.8, 0.2, 0.9, 1.2)])
        surfaces = SurfaceCL([SurfaceCLInfo(0, 10, 0, 1, 0, 1, False),
                              SurfaceCLInfo(0, 10, 0, 1, 0, 1, False)])
        intersection = IntersectionCL(distance=1, position=Vector(0, 0, 0), normal=Vector(0, 0, 1),
                                      surfaceID=0, polygonID=0)
        singleRayDirection = cl.cltypes.make_float3(*rayDirection.array)
        seeds = SeedCL(N)
        fresnelBuffer = FresnelIntersectionCL(N)

        self.program.launchKernel("computeFresnelIntersectionKernel", N=N,
                                  arguments=[singleRayDirection, intersection,
                                             materials, surfaces, seeds, fresnelBuffer])

        fresnelResult = self._getFresnelResult(fresnelBuffer)
        print(fresnelResult)
        self.assertTrue(fresnelResult.isReflected)

    def _getFresnelResult(self, fresnelBuffer) -> FresnelResult:
        fresnelIntersection = self.program.getData(fresnelBuffer)[0]
        incidencePlane = Vector(*fresnelIntersection[:3])
        return FresnelResult(incidencePlane, *fresnelIntersection[4:])

    @staticmethod
    def _getMissingSourceCode():
        with open(os.path.join(OPENCL_SOURCE_DIR, "vectorOperators.c")) as f:
            vectorOperatorsSourceCode = f.read()
        with open(os.path.join(OPENCL_SOURCE_DIR, "random.c")) as f:
            randomSourceCode = f.read()
        return vectorOperatorsSourceCode + randomSourceCode


class IntersectionCL(CLObject):
    STRUCT_NAME = "Intersection"
    STRUCT_DTYPE = np.dtype([("exists", cl.cltypes.uint),
                             ("isTooClose", cl.cltypes.uint),
                             ("distance", cl.cltypes.float),
                             ("position", cl.cltypes.float3),
                             ("normal", cl.cltypes.float3),
                             ("surfaceID", cl.cltypes.uint),
                             ("polygonID", cl.cltypes.uint),
                             ("distanceLeft", cl.cltypes.float)])

    def __init__(self, distance: float, position: Vector, normal: Vector, surfaceID: int, polygonID: int,
                 distanceLeft: float = 0, isTooClose: bool = False):
        self._distance = distance
        self._position = position
        self._normal = normal
        self._surfaceID = surfaceID
        self._polygonID = polygonID
        self._distanceLeft = distanceLeft
        self._isTooClose = isTooClose

        super().__init__(skipDeclaration=False)

    def _getInitialHostBuffer(self) -> np.ndarray:
        buffer = np.empty(1, dtype=self._dtype)
        buffer[0]["exists"] = np.uint32(True)
        buffer[0]["isTooClose"] = np.uint32(self._isTooClose)
        buffer[0]["distance"] = np.float32(self._distance)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[0, 3:6] = self._position.array
        buffer[0, 7:10] = self._normal.array
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        buffer[0]["surfaceID"] = np.uint32(self._surfaceID)
        buffer[0]["polygonID"] = np.uint32(self._polygonID)
        buffer[0]["distanceLeft"] = np.float32(self._distanceLeft)
        return buffer


class FresnelIntersectionCL(CLObject):
    STRUCT_NAME = "FresnelIntersection"
    STRUCT_DTYPE = np.dtype([("incidencePlane", cl.cltypes.float3),
                             ("isReflected", cl.cltypes.uint),
                             ("angleDeflection", cl.cltypes.float),
                             ("nextMaterialID", cl.cltypes.uint),
                             ("nextSolidID", cl.cltypes.int)])

    def __init__(self, N: int):
        self._N = N
        super().__init__(skipDeclaration=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.empty(self._N, dtype=self._dtype)
