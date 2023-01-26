import os
import unittest
from dataclasses import dataclass

import numpy as np

from pytissueoptics import Vector, ScatteringMaterial
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from rayscattering.opencl.CLScene import NO_SURFACE_ID
from rayscattering.opencl.buffers import *

if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None


@dataclass
class PhotonResult:
    position: Vector
    direction: Vector
    er: Vector
    weight: float
    materialID: int
    solidID: int


@dataclass
class DataPointResult:
    deltaWeight: float
    position: Vector
    solidID: int
    surfaceID: int


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLPhoton(unittest.TestCase):
    def setUp(self):
        self.INITIAL_POSITION = Vector(2, 2, 0)
        self.INITIAL_DIRECTION = Vector(0, 0, -1)
        self.INITIAL_WEIGHT = 1.0
        self.INITIAL_SOLID_ID = 9
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "propagation.c")
        self.program = CLProgram(sourcePath)

    def testWhenMoveBy_shouldMovePhotonByTheGivenDistanceTowardsItsDirection(self):
        photonResult = self._photonFunc("moveBy", 10)
        self.assertEqual(self.INITIAL_POSITION + self.INITIAL_DIRECTION * 10, photonResult.position)

    def testWhenScatterByTheta0_shouldNotChangePhotonDirection(self):
        phi, theta = np.pi/4, 0
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorAlmostEqual(self.INITIAL_DIRECTION, photonResult.direction)

    def testWhenScatterByThetaPi_shouldRotatePhotonDirectionToOpposite(self):
        phi, theta = np.pi/4, np.pi

        photonResult = self._photonFunc("scatterBy", phi, theta)

        expectedDirection = self.INITIAL_DIRECTION.copy()
        expectedDirection.multiply(-1)
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction)

    def testWhenScatterByPhi0_shouldNotChangePhotonEr(self):
        photonResult = self._photonFunc("scatterBy", 0, 0)
        initialEr = photonResult.er

        phi, theta = 0, np.pi/4
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorAlmostEqual(initialEr, photonResult.er)

    def testWhenScatterByPhi2Pi_shouldNotChangePhotonEr(self):
        photonResult = self._photonFunc("scatterBy", 0, 0)
        initialEr = photonResult.er

        phi, theta = 2*np.pi, np.pi/4
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorAlmostEqual(initialEr, photonResult.er)

    def testWhenScatterBy_shouldChangePhotonErAndDirection(self):
        photonResult = self._photonFunc("scatterBy", 0, 0)
        initialEr = photonResult.er

        phi, theta = np.pi/4, np.pi/4
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorNotAlmostEqual(initialEr, photonResult.er)
        self._assertVectorNotAlmostEqual(self.INITIAL_DIRECTION, photonResult.direction)

    def testWhenDecreaseWeightBy_shouldDecreaseWeightByTheGivenWeight(self):
        photonResult = self._photonFunc("decreaseWeightBy", 0.2)
        self.assertAlmostEqual(0.8, photonResult.weight)

    def testWhenReflect_shouldReflectPhoton(self):
        self.INITIAL_DIRECTION = Vector(1, -1, 0)
        self.INITIAL_DIRECTION.normalize()
        incidencePlane = Vector(0, 0, 1)
        angleDeflection = np.pi/2

        photonResult = self._photonFunc("reflect", incidencePlane, angleDeflection)

        expectedDirection = Vector(1, 1, 0)
        expectedDirection.normalize()
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction)

    def testWhenRefract_shouldRefractPhoton(self):
        self.INITIAL_DIRECTION = Vector(1, -1, 0)
        self.INITIAL_DIRECTION.normalize()
        incidencePlane = Vector(0, 0, 1)
        angleDeflection = -np.pi/4

        photonResult = self._photonFunc("refract", incidencePlane, angleDeflection)

        expectedDirection = Vector(0, -1, 0)
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction)

    def testWhenRouletteWithWeightAboveThreshold_shouldIgnoreRoulette(self):
        weightThreshold = 1e-4
        self.INITIAL_WEIGHT = weightThreshold * 1.1

        photonResult = self._photonFunc("roulette", weightThreshold, SeedCL(1))

        self.assertAlmostEqual(self.INITIAL_WEIGHT, photonResult.weight)

    def testWhenRouletteWithWeightBelowThresholdAndNotLucky_shouldKillPhoton(self):
        CHANCE = 0.1  # 10% chance of survival, taken from Photon.roulette() implementation
        weightThreshold = 1e-4
        self.INITIAL_WEIGHT = weightThreshold * 0.9
        self._mockRandomValue(CHANCE + 0.01)

        photonResult = self._photonFunc("roulette", weightThreshold, SeedCL(1))

        self.assertAlmostEqual(0, photonResult.weight)

    def testWhenRouletteWithWeightBelowThresholdAndLucky_shouldRescaleWeightToPreserveStatistics(self):
        CHANCE = 0.1
        weightThreshold = 1e-4
        self.INITIAL_WEIGHT = weightThreshold * 0.9
        self._mockRandomValue(CHANCE - 0.01)

        photonResult = self._photonFunc("roulette", weightThreshold, SeedCL(1))

        self.assertAlmostEqual(self.INITIAL_WEIGHT / CHANCE, photonResult.weight)

    def testWhenInteract_shouldDecreasePhotonWeight(self):
        material = ScatteringMaterial(5, 2, 0.9, 1.4)

        photonResult = self._photonFunc("interact", MaterialCL([material]), DataPointCL(1), 0)

        expectedWeightLoss = self.INITIAL_WEIGHT * material.getAlbedo()
        self.assertAlmostEqual(self.INITIAL_WEIGHT - expectedWeightLoss, photonResult.weight)

    def testWhenInteract_shouldLogInteraction(self):
        material = ScatteringMaterial(5, 2, 0.9, 1.4)
        logger = DataPointCL(1)

        self._photonFunc("interact", MaterialCL([material]), logger, 0)
        dataPoint = self._getDataPointResult(logger)

        expectedWeightLoss = self.INITIAL_WEIGHT * material.getAlbedo()
        self.assertAlmostEqual(expectedWeightLoss, dataPoint.deltaWeight)
        self._assertVectorAlmostEqual(self.INITIAL_POSITION, dataPoint.position)
        self.assertEqual(self.INITIAL_SOLID_ID, dataPoint.solidID)
        self.assertEqual(NO_SURFACE_ID, dataPoint.surfaceID)

    def _photonFunc(self, funcName: str, *args) -> PhotonResult:
        self._addMissingDeclarations(args)

        photonBuffer = PhotonCL(positions=np.array([self.INITIAL_POSITION.array]),
                                directions=np.array([self.INITIAL_DIRECTION.array]),
                                materialID=0, solidID=self.INITIAL_SOLID_ID, weight=self.INITIAL_WEIGHT)
        npArgs = [np.float32(arg) if isinstance(arg, (float, int)) else arg for arg in args]
        npArgs = [cl.cltypes.make_float3(*arg.array) if isinstance(arg, Vector) else arg for arg in npArgs]
        self.program.launchKernel(kernelName=funcName + "Kernel", N=1,
                                  arguments=npArgs + [photonBuffer, np.int32(0)])
        photonResult = self._getPhotonResult(photonBuffer)
        return photonResult

    def _addMissingDeclarations(self, kernelArguments):
        self.program._include = ''
        requiredObjects = [MaterialCL([ScatteringMaterial()]), SurfaceCL([]), SeedCL(1), VertexCL([]),
                           DataPointCL(1), SolidCandidateCL(1, 1), TriangleCL([]), SolidCL([])]
        missingObjects = []
        for obj in requiredObjects:
            if any(isinstance(arg, type(obj)) for arg in kernelArguments):
                continue
            missingObjects.append(obj)

        for clObject in missingObjects:
            clObject.make(self.program.device)
            self.program.include(clObject.declaration)

    def _getPhotonResult(self, photonBuffer: PhotonCL):
        data = self.program.getData(photonBuffer)[0]
        return PhotonResult(position=Vector(*data[:3]), direction=Vector(*data[4:7]), er=Vector(*data[8:11]),
                            weight=data[12], materialID=data[13], solidID=data[14])

    def _getDataPointResult(self, dataPointBuffer: DataPointCL):
        data = self.program.getData(dataPointBuffer)[0]
        return DataPointResult(deltaWeight=data[0], position=Vector(*data[1:4]), solidID=data[4], surfaceID=data[5])

    def _assertVectorAlmostEqual(self, v1: Vector, v2: Vector, places=6):
        self.assertAlmostEqual(v1.x, v2.x, places=places)
        self.assertAlmostEqual(v1.y, v2.y, places=places)
        self.assertAlmostEqual(v1.z, v2.z, places=places)

    def _assertVectorNotAlmostEqual(self, v1: Vector, v2: Vector, places=6):
        try:
            self._assertVectorAlmostEqual(v1, v2, places=places)
        except AssertionError:
            return
        self.fail("Vectors are equal")

    def _mockRandomValue(self, value):
        getRandomFloatValueFunction = """float getRandomFloatValue(__global unsigned int *seeds, unsigned int id){
     float result = 0.0f;
     while(result == 0.0f){
         uint rnd_seed = wangHash(seeds[id]);
         seeds[id] = rnd_seed;
         result = (float)rnd_seed / (float)UINT_MAX;
     }
     return result;
}"""
        mockFunction = """float getRandomFloatValue(__global unsigned int *seeds, unsigned int id){
        return %f;
    }""" % value
        self.program.mock(getRandomFloatValueFunction, mockFunction)
