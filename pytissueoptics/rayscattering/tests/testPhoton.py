import math
import sys
import unittest
from unittest.mock import patch

from mockito import mock, when, verify

from pytissueoptics.rayscattering import Photon
from pytissueoptics.rayscattering.photon import WORLD_LABEL, WEIGHT_THRESHOLD
from pytissueoptics.rayscattering.fresnel import FresnelIntersection, FresnelIntersect
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Vector, Logger
from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.intersection.intersectionFinder import Intersection, IntersectionFinder
from pytissueoptics.scene.intersection.mollerTrumboreIntersect import MollerTrumboreIntersect, EPS_CORRECTION
from pytissueoptics.scene.logger import InteractionKey
from pytissueoptics.scene.solids import Solid


EPS = MollerTrumboreIntersect.EPS


class TestPhoton(unittest.TestCase):
    SOLID_INSIDE_LABEL = "solidInside"
    SOLID_OUTSIDE_LABEL = "solidOutside"
    SURFACE_LABEL = "surface"

    def setUp(self):
        self.INITIAL_POSITION = Vector(2, 2, 0)
        self.INITIAL_DIRECTION = Vector(0, 0, -1)
        self.photon = Photon(self.INITIAL_POSITION.copy(), self.INITIAL_DIRECTION.copy())

        self.solidInside = mock(Solid)
        when(self.solidInside).getLabel().thenReturn(self.SOLID_INSIDE_LABEL)
        self.solidOutside = mock(Solid)
        when(self.solidOutside).getLabel().thenReturn(self.SOLID_OUTSIDE_LABEL)

    def testShouldBeInTheGivenState(self):
        self.assertEqual(self.INITIAL_POSITION, self.photon.position)
        self.assertEqual(self.INITIAL_DIRECTION, self.photon.direction)

    def testShouldBeAlive(self):
        self.assertTrue(self.photon.isAlive)

    def testGivenNoContext_whenPropagate_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.photon.propagate()

    def testWhenSetContext_shouldSetInitialMaterial(self):
        material = ScatteringMaterial(mu_s=2, mu_a=1)
        self.photon.setContext(Environment(material))
        self.assertEqual(material, self.photon.material)

    def testWhenMoveBy_shouldMovePhotonByTheGivenDistanceTowardsItsDirection(self):
        self.photon.moveBy(2)
        self.assertEqual(self.INITIAL_POSITION + self.INITIAL_DIRECTION * 2, self.photon.position)

    def testWhenRefract_shouldOrientPhotonTowardsFresnelRefractionAngle(self):
        incidenceAngle = math.pi / 10
        self.photon = Photon(self.INITIAL_POSITION, Vector(0, math.sin(incidenceAngle), -math.cos(incidenceAngle)))
        surfaceNormal = Vector(0, 0, -1)
        incidencePlane = self.photon.direction.cross(surfaceNormal)
        incidencePlane.normalize()
        refractionDeflection = math.pi / 20
        fresnelIntersection = FresnelIntersection(Environment(ScatteringMaterial()), incidencePlane, isReflected=False,
                                                  angleDeflection=refractionDeflection)

        self.photon.refract(fresnelIntersection)

        refractionAngle = incidenceAngle - refractionDeflection
        expectedDirection = Vector(0, math.sin(refractionAngle), -math.cos(refractionAngle))
        self.assertVectorEqual(expectedDirection, self.photon.direction)

    def testWhenScatter_shouldDecreasePhotonWeight(self):
        material = ScatteringMaterial(mu_s=2, mu_a=1)
        self.photon.setContext(Environment(material))

        self.photon.scatter()

        self.assertEqual(1 - material.getAlbedo(), self.photon.weight)

    def testWhenScatter_shouldScatterPhotonDirection(self):
        material = ScatteringMaterial(mu_s=2, mu_a=1)
        theta, phi = math.pi/5, math.pi
        material.getScatteringAngles = lambda: (theta, phi)
        self.photon.setContext(Environment(material))
        self.photon._er = Vector(-1, 0, 0)

        self.photon.scatter()

        expectedDirection = Vector(0, math.sin(theta), -math.cos(theta))
        self.assertVectorEqual(expectedDirection, self.photon.direction)

    def testWhenReflect_shouldOrientPhotonTowardsFresnelReflectionAngle(self):
        incidenceAngle = math.pi / 10
        self.photon = Photon(self.INITIAL_POSITION, Vector(0, math.sin(incidenceAngle), -math.cos(incidenceAngle)))
        surfaceNormal = Vector(0, 0, -1)
        incidencePlane = self.photon.direction.cross(surfaceNormal)
        incidencePlane.normalize()
        reflectionDeflection = 2 * incidenceAngle - math.pi
        fresnelIntersection = FresnelIntersection(Environment(ScatteringMaterial()), incidencePlane, isReflected=True,
                                                  angleDeflection=reflectionDeflection)

        self.photon.reflect(fresnelIntersection)
        reflectionAngle = incidenceAngle - reflectionDeflection
        expectedDirection = Vector(0, math.sin(reflectionAngle), -math.cos(reflectionAngle))
        self.assertVectorEqual(expectedDirection, self.photon.direction)

    def testGivenNoIntersectionFinder_whenPropagate_shouldPropagateUntilItHasNoMoreEnergy(self):
        worldMaterial = ScatteringMaterial(mu_s=2, mu_a=1, g=0.8)
        self.photon.setContext(Environment(worldMaterial))
        initialPosition = self.photon.position.copy()

        self.photon.propagate()

        self.assertFalse(self.photon.isAlive)
        self.assertVectorNotEqual(initialPosition, self.photon.position)

    def testGivenPhotonInVacuum_whenStepWithNoIntersection_shouldKillPhoton(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)

        self.photon.setContext(Environment(ScatteringMaterial()), intersectionFinder=noIntersectionFinder)
        self.photon.step()
        self.assertFalse(self.photon.isAlive)

    def testWhenStepWithIntersection_shouldMovePhotonABitPastIntersection(self):
        """
        The photon starts at z=0, and is moving in the -z direction. The intersection is at z=-8.
        The intersection normal is in the +z direction, so the photon initially lies outside the
        object to intersect. Therefore, we set the photon's context to be in solidOutside.
        We expect photon to intersect at -8 and continue for a distance of EPS_CORRECTION in the
        direction opposite to the intersection normal. In this case since the intersection normal
        is simply +z, we expect the photon to continue towards -z and land at z = -8 - EPS_CORRECTION.
        """
        distance = 8
        intersectionFinder = self._createIntersectionFinder(distance)
        self.photon.setContext(Environment(ScatteringMaterial(), self.solidOutside),
                               intersectionFinder=intersectionFinder)

        self.photon.step(distance + 2)

        expectedDistance = distance + EPS_CORRECTION
        self.assertVectorEqual(self.INITIAL_POSITION + self.photon.direction * expectedDistance,
                               self.photon.position)

    def testWhenStepTooCloseToIntersectionFromOutside_shouldMovePhotonBackABitAndScatter(self):
        distance = 8
        rayLength = distance - 0.5 * EPS
        intersectionFinder = self._createIntersectionFinder(distance, rayLength=rayLength)
        self.photon.setContext(Environment(ScatteringMaterial(), self.solidOutside),
                               intersectionFinder=intersectionFinder)

        self.photon.step(rayLength)

        self.assertVectorEqual(self.INITIAL_POSITION + self.INITIAL_DIRECTION * (rayLength - EPS_CORRECTION),
                               self.photon.position)
        self.assertVectorNotEqual(self.INITIAL_DIRECTION, self.photon.direction)

    def testWhenStepTooCloseToIntersectionFromInside_shouldMovePhotonBackABitAndScatter(self):
        distance = 8
        rayLength = distance - 0.5 * EPS
        intersectionFinder = self._createIntersectionFinder(distance, rayLength=rayLength,
                                                            normal=Vector(0, 0, -1))
        self.photon.setContext(Environment(ScatteringMaterial(), self.solidInside),
                               intersectionFinder=intersectionFinder)

        self.photon.step(rayLength)

        self.assertVectorEqual(self.INITIAL_POSITION + self.INITIAL_DIRECTION * (rayLength - EPS_CORRECTION),
                               self.photon.position)
        self.assertVectorNotEqual(self.INITIAL_DIRECTION, self.photon.direction)

    def testWhenStepWithNoIntersection_shouldMovePhotonAcrossStepDistanceAndScatter(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)
        self.photon.setContext(Environment(ScatteringMaterial(mu_s=2, mu_a=1, g=0.8)), intersectionFinder=noIntersectionFinder)
        stepDistance = 10

        self.photon.step(distance=stepDistance)

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * stepDistance
        self.assertVectorEqual(expectedPosition, self.photon.position)
        self.assertVectorNotEqual(self.INITIAL_DIRECTION, self.photon.direction)

    def testWhenStepWithNoIntersection_shouldReturnADistanceLeftOfZero(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)
        self.photon.setContext(Environment(ScatteringMaterial(mu_s=2, mu_a=1, g=0.8)),
                               intersectionFinder=noIntersectionFinder)

        distanceLeft = self.photon.step()

        self.assertEqual(0, distanceLeft)

    def testWhenStepWithNoDistance_shouldStepWithANewScatteringDistance(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)
        newScatteringDistance = 10
        material = self._createEnvironment(newScatteringDistance)
        self.photon.setContext(material, intersectionFinder=noIntersectionFinder)

        self.photon.step()

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * newScatteringDistance
        self.assertVectorEqual(expectedPosition, self.photon.position)

    def testWhenStepWithReflectingIntersection_shouldMovePhotonABitBackFromIntersection(self):
        totalDistance = 10
        intersectionDistance = 8
        intersectionFinder = self._createIntersectionFinder(intersectionDistance, rayLength=totalDistance)
        environment = self._createEnvironment(scatteringDistance=totalDistance)
        environment.solid = self.solidOutside
        self.photon.setContext(environment, intersectionFinder=intersectionFinder,
                               fresnelIntersect=self._createFresnelIntersectionFactory(isReflected=True))

        self.photon.step(totalDistance)

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * (intersectionDistance - EPS_CORRECTION)
        self.assertVectorEqual(expectedPosition, self.photon.position)

    def testWhenStepWithReflectingIntersection_shouldReturnDistanceLeft(self):
        totalDistance = 10
        intersectionDistance = 8
        intersectionFinder = self._createIntersectionFinder(intersectionDistance, rayLength=totalDistance)
        environment = self._createEnvironment(scatteringDistance=totalDistance)
        environment.solid = self.solidOutside
        self.photon.setContext(environment, intersectionFinder=intersectionFinder,
                               fresnelIntersect=self._createFresnelIntersectionFactory(isReflected=True))

        distanceLeft = self.photon.step(totalDistance)

        expectedDistanceLeft = totalDistance - intersectionDistance - EPS_CORRECTION
        self.assertAlmostEqual(expectedDistanceLeft, distanceLeft)

    def testWhenStepWithRefractingIntersection_shouldUpdatePhotonMaterialToNextMaterial(self):
        nextMaterial = ScatteringMaterial(mu_s=2, mu_a=1, g=0.8)
        self.photon.setContext(Environment(ScatteringMaterial()), intersectionFinder=self._createIntersectionFinder(),
                               fresnelIntersect=self._createFresnelIntersectionFactory(Environment(nextMaterial), isReflected=False))

        self.photon.step()

        self.assertEqual(nextMaterial, self.photon.material)

    def testWhenStepWithRefractingIntersection_shouldMovePhotonABitAfterIntersection(self):
        scatteringDistance = 10
        intersectionDistance = 8
        material = ScatteringMaterial(mu_s=2, mu_a=1, g=0.8)
        nextMaterial = ScatteringMaterial(mu_s=3, mu_a=1, g=0.8)
        self.photon.setContext(Environment(material, self.solidOutside),
                               intersectionFinder=self._createIntersectionFinder(intersectionDistance, scatteringDistance),
                               fresnelIntersect=self._createFresnelIntersectionFactory(Environment(nextMaterial), isReflected=False))

        self.photon.step(scatteringDistance)

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * (intersectionDistance + EPS_CORRECTION)
        self.assertVectorEqual(expectedPosition, self.photon.position)

    def testWhenStepWithRefractingIntersection_shouldReturnDistanceLeftInNextMaterial(self):
        scatteringDistance = 10
        intersectionDistance = 8
        material = ScatteringMaterial(mu_s=2, mu_a=1, g=0.8)
        nextMaterial = ScatteringMaterial(mu_s=3, mu_a=1, g=0.8)
        self.photon.setContext(Environment(material), intersectionFinder=self._createIntersectionFinder(intersectionDistance, scatteringDistance),
                               fresnelIntersect=self._createFresnelIntersectionFactory(Environment(nextMaterial), isReflected=False))

        distanceLeft = self.photon.step(scatteringDistance)

        expectedDistanceLeft = (scatteringDistance - intersectionDistance) * material.mu_t / nextMaterial.mu_t
        expectedDistanceLeft -= EPS_CORRECTION
        self.assertAlmostEqual(expectedDistanceLeft, distanceLeft)

    def testWhenStepWithRefractingIntersectionToVacuum_shouldReturnInfiniteDistanceLeft(self):
        initialScatteringDistance = 10
        intersectionDistance = 8
        environment = Environment(ScatteringMaterial(mu_s=2, mu_a=1, g=0.8))
        nextEnvironment = Environment(ScatteringMaterial(mu_s=0, mu_a=0, g=0))
        self.photon.setContext(environment, intersectionFinder=self._createIntersectionFinder(intersectionDistance, initialScatteringDistance),
                               fresnelIntersect=self._createFresnelIntersectionFactory(nextEnvironment, isReflected=False))

        distanceLeft = self.photon.step(initialScatteringDistance)

        self.assertEqual(math.inf, distanceLeft)

    def testWhenStepWithRefractingIntersectionFromVacuum_shouldReturnZeroDistanceLeft(self):
        initialScatteringDistance = 10
        intersectionDistance = 8
        environment = Environment(ScatteringMaterial(mu_s=0, mu_a=0, g=0))
        nextEnvironment = Environment(ScatteringMaterial(mu_s=2, mu_a=1, g=0.8))
        self.photon.setContext(environment, intersectionFinder=self._createIntersectionFinder(intersectionDistance, initialScatteringDistance),
                               fresnelIntersect=self._createFresnelIntersectionFactory(nextEnvironment, isReflected=False))

        distanceLeft = self.photon.step(initialScatteringDistance)

        self.assertEqual(0, distanceLeft)

    def givenALogger_whenSteppingOutsideASolidAt(self, distance):
        logger = self._createLogger()
        intersectionFinder = self._createIntersectionFinder(distance, normal=self.INITIAL_DIRECTION.copy())
        self.photon.setContext(Environment(ScatteringMaterial()), intersectionFinder=intersectionFinder, logger=logger)

        self.photon.step(distance + 2)
        return logger

    def testGivenALogger_whenSteppingOutsideASolid_shouldLogPositiveWeightOnSurfaceIntersection(self):
        distance = 8
        logger = self.givenALogger_whenSteppingOutsideASolidAt(distance)

        intersectionPoint = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        interactionKey = InteractionKey(self.SOLID_INSIDE_LABEL, self.SURFACE_LABEL)
        verify(logger).logDataPoint(self.photon.weight, intersectionPoint, interactionKey)

    def testGivenALogger_whenSteppingOutsideASolid_shouldLogNegativeWeightOnOtherSolidSurfaceIntersection(self):
        distance = 8
        logger = self.givenALogger_whenSteppingOutsideASolidAt(distance)

        intersectionPoint = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        interactionKey = InteractionKey(self.SOLID_OUTSIDE_LABEL, self.SURFACE_LABEL)
        verify(logger).logDataPoint(-self.photon.weight, intersectionPoint, interactionKey)

    def givenALogger_whenSteppingInsideASolidAt(self, distance):
        logger = self._createLogger()
        enteringSurfaceNormal = self.INITIAL_DIRECTION.copy()
        enteringSurfaceNormal.multiply(-1)
        intersectionFinder = self._createIntersectionFinder(distance, normal=enteringSurfaceNormal)
        self.photon.setContext(Environment(ScatteringMaterial(), self.solidOutside),
                               intersectionFinder=intersectionFinder, logger=logger)

        self.photon.step(distance + 2)
        return logger

    def testGivenALogger_whenSteppingInsideASurface_shouldLogNegativeWeightAtIntersection(self):
        distance = 8
        logger = self.givenALogger_whenSteppingInsideASolidAt(distance)

        intersectionPoint = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        interactionKey = InteractionKey(self.SOLID_INSIDE_LABEL, self.SURFACE_LABEL)
        verify(logger).logDataPoint(-self.photon.weight, intersectionPoint, interactionKey)

    def testGivenALogger_whenSteppingInsideASurface_shouldLogPositiveWeightOnOtherSolidSurfaceIntersection(self):
        distance = 8
        logger = self.givenALogger_whenSteppingInsideASolidAt(distance)

        intersectionPoint = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        interactionKey = InteractionKey(self.SOLID_OUTSIDE_LABEL, self.SURFACE_LABEL)
        verify(logger).logDataPoint(self.photon.weight, intersectionPoint, interactionKey)

    def testGivenALogger_whenScatter_shouldLogWeightLossAtThisPositionWithSolidLabel(self):
        solidInside = mock(Solid)
        when(solidInside).getLabel().thenReturn(self.SOLID_INSIDE_LABEL)
        logger = self._createLogger()
        self.photon.setContext(Environment(ScatteringMaterial(mu_s=3, mu_a=1, g=0.8), solid=solidInside), logger=logger)

        self.photon.scatter()

        weightLoss = self.photon.material.getAlbedo()
        verify(logger).logDataPoint(weightLoss, self.INITIAL_POSITION, InteractionKey(self.SOLID_INSIDE_LABEL))

    def testGivenALogger_whenScatterInWorldMaterial_shouldLogWeightLossAtThisPositionWithWorldLabel(self):
        logger = self._createLogger()
        self.photon.setContext(Environment(ScatteringMaterial(mu_s=3, mu_a=1, g=0.8)), logger=logger)

        self.photon.scatter()

        weightLoss = self.photon.material.getAlbedo()
        verify(logger).logDataPoint(weightLoss, self.INITIAL_POSITION, InteractionKey(WORLD_LABEL))

    def givenALoggerAndNoSolidOutside_whenSteppingInsideASolidAt(self, distance):
        self.solidOutside = None

        logger = self._createLogger()
        enteringSurfaceNormal = self.INITIAL_DIRECTION.copy()
        enteringSurfaceNormal.multiply(-1)
        intersectionFinder = self._createIntersectionFinder(distance, normal=enteringSurfaceNormal)
        self.photon.setContext(Environment(ScatteringMaterial(), self.solidOutside),
                               intersectionFinder=intersectionFinder, logger=logger)

        self.photon.step(distance + 2)
        return logger

    def testGivenALoggerAndNoOutsideSolid_whenSteppingInsideASolid_shouldNotLogWeightOnOutsideSolid(self):
        distance = 8
        logger = self.givenALoggerAndNoSolidOutside_whenSteppingInsideASolidAt(distance)

        intersectionPoint = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        interactionKey = InteractionKey(None, self.SURFACE_LABEL)
        verify(logger, times=1).logDataPoint(...)
        verify(logger, times=0).logDataPoint(self.photon.weight, intersectionPoint, interactionKey)

    def testWhenRouletteWithWeightAboveThreshold_shouldIgnoreRoulette(self):
        self.photon._weight = 1.1 * WEIGHT_THRESHOLD
        self.photon.roulette()

        self.assertTrue(self.photon._weight == 1.1 * WEIGHT_THRESHOLD)

    @patch('random.random', return_value=0.11)
    def testWhenRouletteWithWeightBelowThresholdAndNotLucky_shouldKillPhoton(self, _):
        self.photon._weight = 0.9 * WEIGHT_THRESHOLD
        self.photon.roulette()

        self.assertFalse(self.photon.isAlive)

    @patch('random.random', return_value=0.09)
    def testWhenRouletteWithWeightBelowThresholdAndLucky_shouldRescaleWeightToPreserveStatistics(self, _):
        rouletteChance = 0.1  # defined in Photon.roulette()
        self.photon._weight = 0.9 * WEIGHT_THRESHOLD
        self.photon.roulette()

        self.assertTrue(self.photon._weight == 0.9 * WEIGHT_THRESHOLD / rouletteChance)

    def testWhenInteractWithWeightAtFloatLimit_shouldKillPhoton(self):
        environment = self._createEnvironment(albedo=1.0)
        self.photon.setContext(environment)
        self._weight = sys.float_info.min
        self.assertTrue(self.photon.isAlive)

        self.photon.interact()
        self.assertFalse(self.photon.isAlive)

    def assertVectorEqual(self, v1, v2):
        self.assertAlmostEqual(v1.x, v2.x, places=7)
        self.assertAlmostEqual(v1.y, v2.y, places=7)
        self.assertAlmostEqual(v1.z, v2.z, places=7)

    def assertVectorNotEqual(self, v1, v2):
        self.assertNotAlmostEqual(v1.x, v2.x)
        self.assertNotAlmostEqual(v1.y, v2.y)
        self.assertNotAlmostEqual(v1.z, v2.z)

    def _createIntersectionFinder(self, intersectionDistance=10, rayLength=None, normal=Vector(0, 0, 1)):
        isTooClose = False
        if rayLength is None:
            rayLength = intersectionDistance + 2
        if intersectionDistance and rayLength:
            if abs(intersectionDistance - rayLength) < EPS:
                isTooClose = True
        intersection = Intersection(intersectionDistance, position=Vector(), normal=normal,
                                    insideEnvironment=Environment(ScatteringMaterial(), self.solidInside),
                                    outsideEnvironment=Environment(ScatteringMaterial(), self.solidOutside),
                                    surfaceLabel=self.SURFACE_LABEL, distanceLeft=rayLength-intersectionDistance,
                                    isTooClose=isTooClose)
        intersectionFinder = mock(IntersectionFinder)
        when(intersectionFinder).findIntersection(...).thenReturn(intersection)
        return intersectionFinder

    @staticmethod
    def _createEnvironment(scatteringDistance=1, phi=0.1, theta=0.2, albedo=0.1):
        material = mock(ScatteringMaterial)
        when(material).getScatteringDistance().thenReturn(scatteringDistance)
        when(material).getScatteringAngles().thenReturn((theta, phi))
        when(material).getAlbedo().thenReturn(albedo)
        return Environment(material)

    @staticmethod
    def _createFresnelIntersectionFactory(nextEnvironment=Environment(ScatteringMaterial()),
                                          isReflected=True, angleDeflection=0.1):
        fresnelIntersection = FresnelIntersection(nextEnvironment, Vector(), isReflected=isReflected,
                                                  angleDeflection=angleDeflection)
        fresnelIntersect = mock(FresnelIntersect)
        when(fresnelIntersect).compute(...).thenReturn(fresnelIntersection)
        return fresnelIntersect

    @staticmethod
    def _createLogger():
        logger = mock(Logger)
        when(logger).logPoint(...).thenReturn()
        when(logger).logDataPoint(...).thenReturn()
        return logger
