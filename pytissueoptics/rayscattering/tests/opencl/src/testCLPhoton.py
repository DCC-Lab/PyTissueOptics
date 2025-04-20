import math
import os
import unittest
from dataclasses import dataclass

import numpy as np

from pytissueoptics import ScatteringMaterial, ScatteringScene, Vector
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE, OPENCL_OK
from pytissueoptics.rayscattering.opencl.buffers import (
    DataPointCL,
    MaterialCL,
    PhotonCL,
    SeedCL,
    SolidCandidateCL,
    SolidCL,
    SurfaceCL,
    SurfaceCLInfo,
    TriangleCL,
    TriangleCLInfo,
    VertexCL,
)
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLScene import NO_LOG_ID, NO_SOLID_ID, NO_SURFACE_ID, CLScene
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.scene.geometry import Vertex

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


@unittest.skipIf(not OPENCL_OK, "OpenCL device not available.")
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
        phi, theta = np.pi / 4, 0
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorAlmostEqual(self.INITIAL_DIRECTION, photonResult.direction, places=6)

    def testWhenScatterByThetaPi_shouldRotatePhotonDirectionToOpposite(self):
        phi, theta = np.pi / 4, np.pi

        photonResult = self._photonFunc("scatterBy", phi, theta)

        expectedDirection = self.INITIAL_DIRECTION.copy()
        expectedDirection.multiply(-1)
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction, places=6)

    def testWhenScatterByPhi0_shouldStillResetPhotonEr(self):
        photonResult = self._photonFunc("scatterBy", 0, 0)
        initialEr = photonResult.er

        phi, theta = 0, np.pi / 4
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorNotAlmostEqual(initialEr, photonResult.er)

    def testWhenScatterByPhi2Pi_shouldStillResetPhotonEr(self):
        photonResult = self._photonFunc("scatterBy", 0, 0)
        initialEr = photonResult.er

        phi, theta = 2 * np.pi, np.pi / 4
        photonResult = self._photonFunc("scatterBy", phi, theta)
        self._assertVectorNotAlmostEqual(initialEr, photonResult.er)

    def testWhenScatterBy_shouldChangePhotonErAndDirection(self):
        photonResult = self._photonFunc("scatterBy", 0, 0)
        initialEr = photonResult.er

        phi, theta = np.pi / 4, np.pi / 4
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
        angleDeflection = np.pi / 2

        photonResult = self._photonFunc("reflect", incidencePlane, angleDeflection)

        expectedDirection = Vector(1, 1, 0)
        expectedDirection.normalize()
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction, places=6)

    def testWhenRefract_shouldRefractPhoton(self):
        self.INITIAL_DIRECTION = Vector(1, -1, 0)
        self.INITIAL_DIRECTION.normalize()
        incidencePlane = Vector(0, 0, 1)
        angleDeflection = -np.pi / 4

        photonResult = self._photonFunc("refract", incidencePlane, angleDeflection)

        expectedDirection = Vector(0, -1, 0)
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction, places=6)

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

    def testWhenLogIntersectionWithPhotonLeavingIntoWorld_shouldLogOnlyOneIntersectionOnPreviousSolidWithPositiveWeightCrossing(
        self,
    ):
        self.INITIAL_DIRECTION = Vector(0, 1, 1)
        self.INITIAL_DIRECTION.normalize()
        intersectionNormal = Vector(0, 0, 1)
        insideSolidID = self.INITIAL_SOLID_ID
        outsideSolidID = NO_SOLID_ID

        surfaceID = 0
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID, outsideSolidID, False)])
        logger = DataPointCL(2)

        self._photonFunc("logIntersection", intersectionNormal, 0, surfaces, logger, 0)
        dataPoint1 = self._getDataPointResult(logger, i=0)
        dataPoint2 = self._getDataPointResult(logger, i=1)

        weightCrossing = self.INITIAL_WEIGHT
        self.assertAlmostEqual(weightCrossing, dataPoint1.deltaWeight)
        self._assertVectorAlmostEqual(self.INITIAL_POSITION, dataPoint1.position)
        self.assertEqual(self.INITIAL_SOLID_ID, dataPoint1.solidID)
        self.assertEqual(surfaceID, dataPoint1.surfaceID)

        self.assertAlmostEqual(0, dataPoint2.deltaWeight)
        self.assertAlmostEqual(NO_LOG_ID, dataPoint2.solidID)

    def testWhenLogIntersectionWithPhotonEnteringFromWorld_shouldLogOnlyOneIntersectionOnSolidInsideWithNegativeWeightCrossing(
        self,
    ):
        self.INITIAL_DIRECTION = Vector(0, 1, -1)
        self.INITIAL_DIRECTION.normalize()
        self.INITIAL_SOLID_ID = NO_SOLID_ID
        intersectionNormal = Vector(0, 0, 1)
        insideSolidID = 9
        outsideSolidID = NO_SOLID_ID

        surfaceID = 0
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID, outsideSolidID, False)])
        logger = DataPointCL(2)

        self._photonFunc("logIntersection", intersectionNormal, surfaceID, surfaces, logger, 0)
        dataPoint1 = self._getDataPointResult(logger, i=0)
        dataPoint2 = self._getDataPointResult(logger, i=1)

        weightCrossing = self.INITIAL_WEIGHT
        self.assertAlmostEqual(-weightCrossing, dataPoint1.deltaWeight)
        self._assertVectorAlmostEqual(self.INITIAL_POSITION, dataPoint1.position)
        self.assertEqual(insideSolidID, dataPoint1.solidID)

        self.assertAlmostEqual(0, dataPoint2.deltaWeight)
        self.assertAlmostEqual(NO_LOG_ID, dataPoint2.solidID)

    def testWhenLogIntersectionBetweenTwoSolids_shouldLogIntersectionOnEachSolidWithCorrectWeightCrossings(self):
        # => Photon is leaving solid
        self.INITIAL_DIRECTION = Vector(0, 1, 1)
        self.INITIAL_DIRECTION.normalize()
        intersectionNormal = Vector(0, 0, 1)
        insideSolidID = self.INITIAL_SOLID_ID
        outsideSolidID = self.INITIAL_SOLID_ID + 11

        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID, outsideSolidID, False)])
        logger = DataPointCL(2)

        self._photonFunc("logIntersection", intersectionNormal, 0, surfaces, logger, 0)
        dataPoint1 = self._getDataPointResult(logger, i=0)
        dataPoint2 = self._getDataPointResult(logger, i=1)

        weightCrossing = self.INITIAL_WEIGHT
        self.assertAlmostEqual(weightCrossing, dataPoint1.deltaWeight)
        self.assertEqual(self.INITIAL_SOLID_ID, dataPoint1.solidID)

        self.assertAlmostEqual(-weightCrossing, dataPoint2.deltaWeight)
        self.assertAlmostEqual(outsideSolidID, dataPoint2.solidID)

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

    def testWhenReflectOrRefractWithReflectingIntersection_shouldReflectPhoton(self):
        # => Photon is trying to enter solid
        intersectionNormal = Vector(0, 1, 0)
        self.INITIAL_SOLID_ID = NO_SOLID_ID
        self.INITIAL_DIRECTION = Vector(1, -1, 0)
        self.INITIAL_DIRECTION.normalize()
        self._mockFresnelIntersection(isReflected=True, incidencePlane=Vector(0, 0, 1), angleDeflection=np.pi / 2)

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID=9, outsideSolidID=NO_SOLID_ID, toSmooth=False)])
        photonResult = self._photonFunc(
            "reflectOrRefract",
            intersectionNormal,
            0,
            10,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            logger,
            0,
            SeedCL(1),
        )

        expectedDirection = Vector(1, 1, 0)
        expectedDirection.normalize()
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction, places=6)
        expectedPosition = self.INITIAL_POSITION
        self._assertVectorAlmostEqual(expectedPosition, photonResult.position)
        self.assertEqual(NO_SOLID_ID, photonResult.solidID)

        dataPoint = self._getDataPointResult(logger)
        self.assertEqual(NO_LOG_ID, dataPoint.solidID)

    def testWhenReflectOrRefractWithRefractingIntersection_shouldRefractPhoton(self):
        # => Photon enters solid
        intersectionNormal = Vector(0, 1, 0)
        self.INITIAL_SOLID_ID = NO_SOLID_ID
        self.INITIAL_DIRECTION = Vector(1, -1, 0)
        self.INITIAL_DIRECTION.normalize()
        insideSolidID = 9
        insideMaterialID = 0
        self._mockFresnelIntersection(
            isReflected=False,
            incidencePlane=Vector(0, 0, 1),
            angleDeflection=-np.pi / 4,
            nextMaterialID=insideMaterialID,
            nextSolidID=insideSolidID,
        )

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID, outsideSolidID=NO_SOLID_ID, toSmooth=False)])
        photonResult = self._photonFunc(
            "reflectOrRefract",
            intersectionNormal,
            0,
            10,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            logger,
            0,
            SeedCL(1),
        )

        expectedDirection = Vector(0, -1, 0)
        expectedDirection.normalize()
        self._assertVectorAlmostEqual(expectedDirection, photonResult.direction, places=6)
        expectedPosition = self.INITIAL_POSITION
        self._assertVectorAlmostEqual(expectedPosition, photonResult.position)
        self.assertEqual(insideSolidID, photonResult.solidID)
        self.assertEqual(insideMaterialID, photonResult.materialID)

        # (not a unit test at this point, but)
        # Should also log the entering intersection with negative weight crossing
        dataPoint = self._getDataPointResult(logger)
        self.assertEqual(-self.INITIAL_WEIGHT, dataPoint.deltaWeight)
        self._assertVectorAlmostEqual(self.INITIAL_POSITION, dataPoint.position)
        self.assertEqual(insideSolidID, dataPoint.solidID)

    def testWhenStepToInfinityWithNoIntersection_shouldKillPhoton(self):
        stepDistance = math.inf
        self._mockFindIntersection(exists=False)

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID=9, outsideSolidID=10, toSmooth=False)])
        photonResult = self._photonFunc(
            "propagateStep",
            stepDistance,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            TriangleCL([]),
            VertexCL([]),
            SeedCL(1),
            logger,
            0,
        )

        self.assertEqual(0, photonResult.weight)

    def testWhenStepWithNoIntersection_shouldMovePhotonAndScatter(self):
        stepDistance = 10
        self._mockFindIntersection(exists=False)

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID=9, outsideSolidID=10, toSmooth=False)])
        photonResult = self._photonFunc(
            "propagateStep",
            stepDistance,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            TriangleCL([]),
            VertexCL([]),
            SeedCL(1),
            logger,
            0,
        )

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * stepDistance
        self._assertVectorAlmostEqual(expectedPosition, photonResult.position)
        interaction = self._getDataPointResult(logger)
        self.assertEqual(self.INITIAL_SOLID_ID, interaction.solidID)

    def testWhenStepWithIntersectionTooCloseToVertex_shouldMovePhotonAwayABitAndScatter(self):
        stepDistance = 10.0
        nextSolidID = self.INITIAL_SOLID_ID + 1
        self.INITIAL_DIRECTION = Vector(0, 0, -1)
        normal = Vector(0, 0, 1)
        self._mockFindIntersection(exists=True, distance=stepDistance, normal=normal)
        intersectionPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * stepDistance

        logger = DataPointCL(2)
        surfaces = SurfaceCL(
            [SurfaceCLInfo(0, 0, 0, 0, insideSolidID=nextSolidID, outsideSolidID=self.INITIAL_SOLID_ID, toSmooth=False)]
        )
        triangles = TriangleCL([TriangleCLInfo([0, 1, 2], normal)])
        # Put vertex slightly after the intersection point
        vertex = Vertex(intersectionPosition.x, intersectionPosition.y, intersectionPosition.z - 2e-7)
        vertex.normal = normal
        photonResult = self._photonFunc(
            "propagateStep",
            stepDistance,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            triangles,
            VertexCL([vertex]),
            SeedCL(1),
            logger,
            0,
        )

        # Due to the correction being under floating point rounding error, we cannot measure it.
        self._assertVectorAlmostEqual(intersectionPosition, photonResult.position)
        interaction = self._getDataPointResult(logger)
        self.assertEqual(nextSolidID, interaction.solidID)

    def testWhenStepWithNoDistance_shouldStepWithANewScatteringDistance(self):
        stepDistance = 0
        self._mockFindIntersection(exists=False)

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID=9, outsideSolidID=10, toSmooth=False)])
        photonResult = self._photonFunc(
            "propagateStep",
            stepDistance,
            MaterialCL([ScatteringMaterial(5, 2)]),
            surfaces,
            TriangleCL([]),
            VertexCL([]),
            SeedCL(1),
            logger,
            0,
        )

        self._assertVectorNotAlmostEqual(self.INITIAL_POSITION, photonResult.position)

    def testWhenStepWithIntersectionReflecting_shouldMovePhotonToIntersection(self):
        stepDistance = 10
        intersectionDistance = 5
        self._mockFindIntersection(exists=True, distance=intersectionDistance)
        self._mockFresnelIntersection(isReflected=True)

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID=9, outsideSolidID=10, toSmooth=False)])
        triangles = TriangleCL([TriangleCLInfo([0, 1, 2], Vector(0, 0, 1))])
        vertices = VertexCL([Vertex(0, 0, 0)] * 3)
        photonResult = self._photonFunc(
            "propagateStep",
            stepDistance,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            triangles,
            vertices,
            SeedCL(1),
            logger,
            0,
        )

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * intersectionDistance
        self._assertVectorAlmostEqual(expectedPosition, photonResult.position)

    def testWhenStepWithIntersectionRefracting_shouldMovePhotonToIntersection(self):
        stepDistance = 10
        intersectionDistance = 5
        self._mockFindIntersection(exists=True, distance=intersectionDistance)
        self._mockFresnelIntersection(isReflected=False)

        logger = DataPointCL(2)
        surfaces = SurfaceCL([SurfaceCLInfo(0, 0, 0, 0, insideSolidID=9, outsideSolidID=10, toSmooth=False)])
        triangles = TriangleCL([TriangleCLInfo([0, 1, 2], Vector(0, 0, 1))])
        vertices = VertexCL([Vertex(0, 0, 0)] * 3)
        photonResult = self._photonFunc(
            "propagateStep",
            stepDistance,
            MaterialCL([ScatteringMaterial()]),
            surfaces,
            triangles,
            vertices,
            SeedCL(1),
            logger,
            0,
        )

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * intersectionDistance
        self._assertVectorAlmostEqual(expectedPosition, photonResult.position)

    def testWhenPropagateInInfiniteMedium_shouldPropagateUntilItHasNoMoreEnergy(self):
        self._mockFindIntersection(exists=False)
        rouletteChance = 0.1  # taken from the code
        self._mockRandomValue(rouletteChance * 2)  # deactivates roulette rescaling

        photonResult = self._photonPropagateInInfiniteMedium()

        self._assertVectorNotAlmostEqual(self.INITIAL_POSITION, photonResult.position)
        self.assertEqual(0, photonResult.weight)

    def testWhenPropagateReachesMaxInteractions_shouldReturnCurrentPhotonState(self):
        self._mockFindIntersection(exists=False)
        self._mockRandomValue(0.2)  # deactivates roulette rescaling

        photonResult = self._photonPropagateInInfiniteMedium(factorOfMaxInteractions=0.5)

        self._assertVectorNotAlmostEqual(self.INITIAL_POSITION, photonResult.position)
        self.assertNotEqual(self.INITIAL_WEIGHT, photonResult.weight)
        self.assertNotEqual(0, photonResult.weight)

    def _photonFunc(self, funcName: str, *args) -> PhotonResult:
        self._addMissingDeclarations(args)

        photonBuffer = PhotonCL(
            positions=np.array([self.INITIAL_POSITION.array]),
            directions=np.array([self.INITIAL_DIRECTION.array]),
            materialID=0,
            solidID=self.INITIAL_SOLID_ID,
            weight=self.INITIAL_WEIGHT,
        )
        npArgs = [np.float32(arg) if isinstance(arg, (float, int)) else arg for arg in args]
        npArgs = [cl.cltypes.make_float3(*arg.array) if isinstance(arg, Vector) else arg for arg in npArgs]
        self.program.launchKernel(kernelName=funcName + "Kernel", N=1, arguments=npArgs + [photonBuffer, np.int32(0)])
        photonResult = self._getPhotonResult(photonBuffer)
        return photonResult

    def _photonPropagateInInfiniteMedium(self, factorOfMaxInteractions=1.0) -> PhotonResult:
        material = ScatteringMaterial(5, 2, 0.9, 1.4)
        WEIGHT_THRESHOLD = 0.02
        # With roulette rescaling OFF, this is a bit more than the number of interactions needed to reach the threshold
        avgInteractions = -np.log(WEIGHT_THRESHOLD) / material.getAlbedo()
        maxInteractions = int(np.ceil(avgInteractions) * factorOfMaxInteractions)

        s = self._getCLSceneOfInfiniteMedium(material)
        logger = DataPointCL(maxInteractions)
        photonBuffer = PhotonCL(
            positions=np.array([self.INITIAL_POSITION.array]),
            directions=np.array([self.INITIAL_DIRECTION.array]),
            materialID=0,
            solidID=self.INITIAL_SOLID_ID,
            weight=self.INITIAL_WEIGHT,
        )
        self.program.launchKernel(
            kernelName="propagate",
            N=1,
            arguments=[
                np.int32(1),
                np.int32(maxInteractions),
                np.float32(WEIGHT_THRESHOLD),
                np.int32(1),
                photonBuffer,
                s.materials,
                s.nSolids,
                s.solids,
                s.surfaces,
                s.triangles,
                s.vertices,
                s.solidCandidates,
                SeedCL(1),
                logger,
            ],
        )
        return self._getPhotonResult(photonBuffer)

    @staticmethod
    def _getCLSceneOfInfiniteMedium(material):
        scene = ScatteringScene([], worldMaterial=material)
        sceneCL = CLScene(scene, nWorkUnits=1)
        return sceneCL

    def _addMissingDeclarations(self, kernelArguments):
        self.program._include = ""
        requiredObjects = [
            MaterialCL([ScatteringMaterial()]),
            SurfaceCL([]),
            SeedCL(1),
            VertexCL([]),
            DataPointCL(1),
            SolidCandidateCL(1, 1),
            TriangleCL([]),
            SolidCL([]),
        ]
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
        return PhotonResult(
            position=Vector(*data[:3]),
            direction=Vector(*data[4:7]),
            er=Vector(*data[8:11]),
            weight=data[12],
            materialID=data[13],
            solidID=data[14],
        )

    def _getDataPointResult(self, dataPointBuffer: DataPointCL, i=0):
        data = self.program.getData(dataPointBuffer)[i]
        return DataPointResult(deltaWeight=data[0], position=Vector(*data[1:4]), solidID=data[4], surfaceID=data[5])

    def _assertVectorAlmostEqual(self, v1: Vector, v2: Vector, places=7):
        self.assertAlmostEqual(v1.x, v2.x, places=places)
        self.assertAlmostEqual(v1.y, v2.y, places=places)
        self.assertAlmostEqual(v1.z, v2.z, places=places)

    def _assertVectorNotAlmostEqual(self, v1: Vector, v2: Vector, places=7):
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
        mockFunction = (
            """float getRandomFloatValue(__global unsigned int *seeds, unsigned int id){
        return %f;
    }"""
            % value
        )
        self.program.mock(getRandomFloatValueFunction, mockFunction)

    def _mockFresnelIntersection(
        self,
        isReflected: bool,
        incidencePlane: Vector = Vector(0, 1, 0),
        angleDeflection: float = np.pi / 2,
        nextMaterialID=0,
        nextSolidID=0,
    ):
        fresnelCall = """FresnelIntersection fresnelIntersection = computeFresnelIntersection(photons[photonID].direction, intersection,
                                                                         materials, surfaces, seeds, gid);"""
        x, y, z = incidencePlane.array
        mockCall = """FresnelIntersection fresnelIntersection;
        fresnelIntersection.isReflected = %s;
        fresnelIntersection.incidencePlane = (float3)(%f, %f, %f);
        fresnelIntersection.angleDeflection = %f;
        fresnelIntersection.nextMaterialID = %d;
        fresnelIntersection.nextSolidID = %d;
        """ % (str(isReflected).lower(), x, y, z, angleDeflection, nextMaterialID, nextSolidID)
        self.program.mock(fresnelCall, mockCall)

    def _mockFindIntersection(self, exists=True, distance=8.0, normal=Vector(0, 0, 1), surfaceID=0, distanceLeft=2):
        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        intersectionCall = (
            """Intersection intersection = findIntersection(stepRay, scene, gid, photons[photonID].solidID);"""
        )
        px, py, pz = expectedPosition.array
        nx, ny, nz = normal.array
        mockCall = """Intersection intersection;
        intersection.exists = %s;
        intersection.distance = %.7f;
        intersection.position = (float3)(%.7f, %.7f, %.7f);
        intersection.normal = (float3)(%.f, %f, %f);
        intersection.surfaceID = %d;
        intersection.distanceLeft = %f;
        """ % (str(exists).lower(), distance, px, py, pz, nx, ny, nz, surfaceID, distanceLeft)
        self.program.mock(intersectionCall, mockCall)
