import math
import os
import unittest
from dataclasses import dataclass

import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics import Vector, ScatteringMaterial
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.buffers import MaterialCL, SurfaceCL, SurfaceCLInfo, SeedCL


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
    OUTSIDE_SOLID_ID = 0
    INSIDE_SOLID_ID = 1
    OUTSIDE_MATERIAL_ID = 0
    INSIDE_MATERIAL_ID = 1

    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "fresnel.c")
        self.program = CLProgram(sourcePath)
        self.program.include(self._getMissingSourceCode())

    def _setUpWith(self, n1=1.0, n2=1.5, normal=Vector(0, 0, 1)):
        outsideMaterial = ScatteringMaterial(0.8, 0.2, 0.9, n1)
        insideMaterial = ScatteringMaterial(0.8, 0.2, 0.9, n2)
        self.materials = MaterialCL([outsideMaterial, insideMaterial])
        self.surfaces = SurfaceCL([SurfaceCLInfo(0, 1, self.INSIDE_MATERIAL_ID, self.OUTSIDE_MATERIAL_ID,
                                                 self.INSIDE_SOLID_ID, self.OUTSIDE_SOLID_ID, False)])
        self.normal = normal
        self.n1 = n1
        self.n2 = n2
        self.intersection = IntersectionCL(normal=normal, surfaceID=0)
        self.rayAt45 = Vector(1, 0, -1)
        self.rayAt45.normalize()

    def testWhenCompute_shouldReturnAFresnelIntersection(self):
        self._setUpWith(n1=1.0, n2=1.5)

        fresnelResult = self._computeFresnelIntersection(self.rayAt45)

        self.assertNotEqual(0, fresnelResult.angleDeflection)
        self.assertEqual(Vector(0, 1, 0), fresnelResult.incidencePlane)

    def testWhenIsReflected_shouldComputeReflectionDeflection(self):
        self._setUpWith(n1=1.0, n2=1.5)
        self._mockIsReflected(True)

        fresnelResult = self._computeFresnelIntersection(self.rayAt45)

        self.assertTrue(fresnelResult.isReflected)
        self.assertAlmostEqual(-np.pi / 2, fresnelResult.angleDeflection, places=6)

    def testWhenIsRefracted_shouldComputeRefractionDeflection(self):
        self._setUpWith(n1=1.0, n2=1.5)
        self._mockIsReflected(False)

        fresnelResult = self._computeFresnelIntersection(self.rayAt45)

        expectedDeflection = np.pi / 4 - np.arcsin(self.n1 / self.n2 * np.sin(np.pi / 4))
        self.assertFalse(fresnelResult.isReflected)
        self.assertAlmostEqual(expectedDeflection, fresnelResult.angleDeflection, places=6)

    def testIfGoingInside_shouldSetNextEnvironmentAsEnvironmentInside(self):
        self._setUpWith(n1=1.0, n2=1.5)
        self._mockIsReflected(False)

        fresnelResult = self._computeFresnelIntersection(self.rayAt45)

        self.assertEqual(self.INSIDE_MATERIAL_ID, fresnelResult.nextMaterialID)
        self.assertEqual(self.INSIDE_SOLID_ID, fresnelResult.nextSolidID)

    def testIfGoingOutside_shouldSetNextEnvironmentAsEnvironmentOutside(self):
        surfaceNormalAlongRayDirection = Vector(0, 0, -1)
        self._setUpWith(n1=1.0, n2=1.5, normal=surfaceNormalAlongRayDirection)
        self._mockIsReflected(False)

        fresnelResult = self._computeFresnelIntersection(self.rayAt45)

        self.assertEqual(self.OUTSIDE_MATERIAL_ID, fresnelResult.nextMaterialID)
        self.assertEqual(self.OUTSIDE_SOLID_ID, fresnelResult.nextSolidID)

    def testGivenAnAngleOfIncidenceAboveTotalInternalReflection_shouldHaveAReflectionCoefficientOf1(self):
        self._setUpWith(n1=np.sqrt(2.01), n2=1)
        R = self._getReflectionCoefficient(self.rayAt45)
        self.assertEqual(1, R)

    def testGivenSameRefractiveIndices_shouldHaveAReflectionCoefficientOf0(self):
        self._setUpWith(n1=1.5, n2=1.5)
        R = self._getReflectionCoefficient(self.rayAt45)
        self.assertEqual(0, R)

    def testGivenPerpendicularIncidence_shouldHaveCorrectReflectionCoefficient(self):
        self._setUpWith(n1=1.0, n2=1.5, normal=Vector(0, 0, 1))

        R = self._getReflectionCoefficient(rayDirection=Vector(0, 0, -1))

        expectedR = ((self.n2 - self.n1) / (self.n2 + self.n1)) ** 2
        self.assertAlmostEqual(expectedR, R, places=6)

    def _computeFresnelIntersection(self, rayDirection: Vector) -> FresnelResult:
        N = 1  # Kernel size errors when trying a vector buffer. Limiting to 1 for now, which is fine for testing.
        singleRayDirectionBuffer = cl.cltypes.make_float3(*rayDirection.array)
        seeds = SeedCL(N)
        fresnelBuffer = FresnelIntersectionCL(N)
        self.program.launchKernel("computeFresnelIntersectionKernel", N=N,
                                  arguments=[singleRayDirectionBuffer, self.intersection,
                                             self.materials, self.surfaces, seeds, fresnelBuffer])

        return self._getFresnelResult(fresnelBuffer)

    def _getReflectionCoefficient(self, rayDirection: Vector) -> float:
        normal = self.normal.copy()
        goingInside = rayDirection.dot(normal) < 0
        if goingInside:
            normal.multiply(-1)
        thetaIn = math.acos(normal.dot(rayDirection))

        N = 1
        coefficientBuffer = FloatContainerCL(N)

        self.program._build([self.materials, self.surfaces, self.intersection])
        self.program.include(self.materials.declaration + self.surfaces.declaration + self.intersection.declaration)

        self.program.launchKernel("getReflectionCoefficientKernel", N=N,
                                  arguments=[np.float32(self.n1), np.float32(self.n2),
                                             np.float32(thetaIn), coefficientBuffer])
        return float(self.program.getData(coefficientBuffer)[0])

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

    def _mockIsReflected(self, isReflected: bool):
        isReflectedFunction = """bool _getIsReflected(float nIn, float nOut, float thetaIn, __global uint *seeds, uint gid) {
    float R = _getReflectionCoefficient(nIn, nOut, thetaIn);
    float randomFloat = getRandomFloatValue(seeds, gid);
    if (R > randomFloat) {
        return true;
    }
    return false;
}"""
        mockFunction = """bool _getIsReflected(float nIn, float nOut, float thetaIn, __global uint *seeds, uint gid) {
        return %s;
        }""" % str(isReflected).lower()
        self.program.mock(isReflectedFunction, mockFunction)


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

    def __init__(self, distance: float = 10, position=Vector(0, 0, 0), normal=Vector(0, 0, 1),
                 surfaceID=0, polygonID=0, distanceLeft: float = 0, isTooClose: bool = False):
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


class FloatContainerCL(CLObject):
    STRUCT_NAME = "FloatContainer"
    STRUCT_DTYPE = np.dtype([("value", cl.cltypes.float)])

    def __init__(self, N: int):
        self._N = N
        super().__init__(skipDeclaration=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.empty(self._N, dtype=self._dtype)



if __name__ == "__main__":
    unittest.main()
