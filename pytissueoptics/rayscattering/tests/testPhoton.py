import math
import sys
import unittest

from mockito import mock, when, verify

from pytissueoptics.rayscattering import Photon
from pytissueoptics.rayscattering.fresnel import FresnelIntersection, FresnelIntersectionFactory
from pytissueoptics.scene import Vector, Material, Logger
from pytissueoptics.scene.geometry import Polygon
from pytissueoptics.scene.intersection.intersectionFinder import Intersection, IntersectionFinder


class TestPhoton(unittest.TestCase):
    def setUp(self):
        self.INITIAL_POSITION = Vector(2, 2, 0)
        self.INITIAL_DIRECTION = Vector(0, 0, -1)
        self.photon = Photon(self.INITIAL_POSITION.copy(), self.INITIAL_DIRECTION.copy())

    def testShouldBeInTheGivenState(self):
        self.assertEqual(self.INITIAL_POSITION, self.photon.position)
        self.assertEqual(self.INITIAL_DIRECTION, self.photon.direction)

    def testShouldBeAlive(self):
        self.assertTrue(self.photon.isAlive)

    def testGivenNoContext_whenPropagate_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.photon.propagate()

    def testWhenSetContext_shouldSetInitialMaterial(self):
        material = Material(mu_s=2, mu_a=1)
        self.photon.setContext(material)
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
        fresnelIntersection = FresnelIntersection(Material(), incidencePlane, isReflected=False,
                                                  angleDeflection=refractionDeflection)

        self.photon.refract(fresnelIntersection)

        refractionAngle = incidenceAngle - refractionDeflection
        expectedDirection = Vector(0, math.sin(refractionAngle), -math.cos(refractionAngle))
        self.assertVectorEqual(expectedDirection, self.photon.direction)

    def testWhenScatter_shouldDecreasePhotonWeight(self):
        material = Material(mu_s=2, mu_a=1)
        self.photon.setContext(material)

        self.photon.scatter()

        self.assertEqual(1 - material.getAlbedo(), self.photon.weight)

    def testWhenScatter_shouldScatterPhotonDirection(self):
        material = Material(mu_s=2, mu_a=1)
        theta, phi = math.pi/5, math.pi
        material.getScatteringAngles = lambda: (theta, phi)
        self.photon.setContext(material)
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
        fresnelIntersection = FresnelIntersection(Material(), incidencePlane, isReflected=True,
                                                  angleDeflection=reflectionDeflection)

        self.photon.reflect(fresnelIntersection)
        reflectionAngle = incidenceAngle - reflectionDeflection
        expectedDirection = Vector(0, math.sin(reflectionAngle), -math.cos(reflectionAngle))
        self.assertVectorEqual(expectedDirection, self.photon.direction)

    def testGivenNoIntersectionFinder_whenPropagate_shouldPropagateUntilItHasNoMoreEnergy(self):
        worldMaterial = Material(mu_s=2, mu_a=1, g=0.8)
        self.photon.setContext(worldMaterial)
        initialPosition = self.photon.position.copy()

        self.photon.propagate()

        self.assertFalse(self.photon.isAlive)
        self.assertVectorNotEqual(initialPosition, self.photon.position)

    def testGivenPhotonInVacuum_whenStepWithNoIntersection_shouldKillPhoton(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)

        self.photon.setContext(Material(), intersectionFinder=noIntersectionFinder)
        self.photon.step()
        self.assertFalse(self.photon.isAlive)

    def testWhenStepWithIntersection_shouldMovePhotonToIntersection(self):
        distance = 8
        intersectionFinder = self._createIntersectionFinder(distance)
        self.photon.setContext(Material(), intersectionFinder=intersectionFinder)

        self.photon.step(distance + 2)

        self.assertVectorEqual(self.INITIAL_POSITION + self.photon.direction * distance,
                               self.photon.position)

    def testWhenStepWithNoIntersection_shouldMovePhotonAcrossStepDistanceAndScatter(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)
        self.photon.setContext(Material(mu_s=2, mu_a=1, g=0.8), intersectionFinder=noIntersectionFinder)
        stepDistance = 10

        self.photon.step(distance=stepDistance)

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * stepDistance
        self.assertVectorEqual(expectedPosition, self.photon.position)
        self.assertVectorNotEqual(self.INITIAL_DIRECTION, self.photon.direction)

    def testWhenStepWithNoDistance_shouldStepWithANewScatteringDistance(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)
        newScatteringDistance = 10
        material = self._createMaterial(newScatteringDistance)
        self.photon.setContext(material, intersectionFinder=noIntersectionFinder)

        self.photon.step()

        expectedPosition = self.INITIAL_POSITION + self.INITIAL_DIRECTION * newScatteringDistance
        self.assertVectorEqual(expectedPosition, self.photon.position)

    def testWhenStepWithNoIntersection_shouldReturnADistanceLeftOfZero(self):
        noIntersectionFinder = mock(IntersectionFinder)
        when(noIntersectionFinder).findIntersection(...).thenReturn(None)
        self.photon.setContext(Material(mu_s=2, mu_a=1, g=0.8), intersectionFinder=noIntersectionFinder)

        distanceLeft = self.photon.step()

        self.assertEqual(0, distanceLeft)

    def testWhenStepWithReflectingIntersection_shouldReturnDistanceLeft(self):
        totalDistance = 10
        intersectionDistance = 8
        intersectionFinder = self._createIntersectionFinder(intersectionDistance, totalDistance)
        material = self._createMaterial(scatteringDistance=totalDistance)
        self.photon.setContext(material, intersectionFinder=intersectionFinder,
                               fresnelIntersectionFactory=self._createFresnelIntersectionFactory(isReflected=True))

        distanceLeft = self.photon.step(totalDistance)

        self.assertEqual(totalDistance - intersectionDistance, distanceLeft)

    def testWhenStepWithRefractingIntersection_shouldUpdatePhotonMaterialToNextMaterial(self):
        nextMaterial = Material(mu_s=2, mu_a=1, g=0.8)
        self.photon.setContext(Material(), intersectionFinder=self._createIntersectionFinder(),
                               fresnelIntersectionFactory=self._createFresnelIntersectionFactory(nextMaterial, isReflected=False))

        self.photon.step()

        self.assertEqual(nextMaterial, self.photon.material)

    def testWhenStepWithRefractingIntersection_shouldReturnDistanceLeftInNextMaterial(self):
        initialScatteringDistance = 10
        intersectionDistance = 8
        material = Material(mu_s=2, mu_a=1, g=0.8)
        nextMaterial = Material(mu_s=3, mu_a=1, g=0.8)
        self.photon.setContext(material, intersectionFinder=self._createIntersectionFinder(intersectionDistance, initialScatteringDistance),
                               fresnelIntersectionFactory=self._createFresnelIntersectionFactory(nextMaterial, isReflected=False))

        distanceLeft = self.photon.step(initialScatteringDistance)

        expectedDistanceLeft = (initialScatteringDistance - intersectionDistance) * material.mu_t / nextMaterial.mu_t
        self.assertEqual(expectedDistanceLeft, distanceLeft)

    def testWhenStepWithRefractingIntersectionToVacuum_shouldReturnInfiniteDistanceLeft(self):
        initialScatteringDistance = 10
        intersectionDistance = 8
        material = Material(mu_s=2, mu_a=1, g=0.8)
        nextMaterial = Material(mu_s=0, mu_a=0, g=0)
        self.photon.setContext(material, intersectionFinder=self._createIntersectionFinder(intersectionDistance, initialScatteringDistance),
                               fresnelIntersectionFactory=self._createFresnelIntersectionFactory(nextMaterial, isReflected=False))

        distanceLeft = self.photon.step(initialScatteringDistance)

        self.assertEqual(math.inf, distanceLeft)

    def testWhenStepWithRefractingIntersectionFromVacuum_shouldReturnZeroDistanceLeft(self):
        initialScatteringDistance = 10
        intersectionDistance = 8
        material = Material(mu_s=0, mu_a=0, g=0)
        nextMaterial = Material(mu_s=2, mu_a=1, g=0.8)
        self.photon.setContext(material, intersectionFinder=self._createIntersectionFinder(intersectionDistance, initialScatteringDistance),
                               fresnelIntersectionFactory=self._createFresnelIntersectionFactory(nextMaterial, isReflected=False))

        distanceLeft = self.photon.step(initialScatteringDistance)

        self.assertEqual(0, distanceLeft)

    def testGivenALogger_whenPropagate_shouldLogInitialPosition(self):
        logger = self._createLogger()
        self.photon.setContext(Material(mu_s=2, mu_a=1, g=0.8), logger=logger)

        self.photon.propagate()

        verify(logger).logPoint(self.INITIAL_POSITION)

    def testGivenALogger_whenStep_shouldLogIntersectionPositions(self):
        logger = self._createLogger()
        distance = 8
        intersectionFinder = self._createIntersectionFinder(distance)
        self.photon.setContext(Material(), intersectionFinder=intersectionFinder, logger=logger)

        self.photon.step(distance+2)

        intersectionPoint = self.INITIAL_POSITION + self.INITIAL_DIRECTION * distance
        verify(logger).logPoint(intersectionPoint)

    def testGivenALogger_whenScatter_shouldLogWeightLossAtThisPosition(self):
        logger = self._createLogger()
        self.photon.setContext(Material(mu_s=3, mu_a=1, g=0.8), logger=logger)

        self.photon.scatter()

        weightLoss = self.photon.material.getAlbedo()
        verify(logger).logDataPoint(weightLoss, position=self.INITIAL_POSITION)

    def testInteractAtFloatLimitIsStillValid(self):
        material = self._createMaterial(albedo=1.0)
        self.photon.setContext(material)
        self._weight = sys.float_info.min
        self.assertTrue(self.photon.isAlive)

        self.photon.interact()
        self.assertFalse(self.photon.isAlive)

    def assertVectorEqual(self, v1, v2):
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertAlmostEqual(v1.z, v2.z)

    def assertVectorNotEqual(self, v1, v2):
        self.assertNotAlmostEqual(v1.x, v2.x)
        self.assertNotAlmostEqual(v1.y, v2.y)
        self.assertNotAlmostEqual(v1.z, v2.z)

    @staticmethod
    def _createIntersectionFinder(intersectionDistance=10, rayLength=None):
        if rayLength is None:
            rayLength = intersectionDistance + 2
        aPolygon = Polygon([Vector(), Vector(), Vector()],
                           insideMaterial=Material(), outsideMaterial=Material())
        intersection = Intersection(intersectionDistance, position=Vector(), polygon=aPolygon,
                                    distanceLeft=rayLength-intersectionDistance)
        intersectionFinder = mock(IntersectionFinder)
        when(intersectionFinder).findIntersection(...).thenReturn(intersection)
        return intersectionFinder

    @staticmethod
    def _createMaterial(scatteringDistance=1, phi=0.1, theta=0.2, albedo=0.1):
        material = mock(Material)
        when(material).getScatteringDistance().thenReturn(scatteringDistance)
        when(material).getScatteringAngles().thenReturn((theta, phi))
        when(material).getAlbedo().thenReturn(albedo)
        return material

    @staticmethod
    def _createFresnelIntersectionFactory(nextMaterial=Material(), isReflected=True, angleDeflection=0.1):
        fresnelIntersection = FresnelIntersection(nextMaterial, Vector(), isReflected=isReflected,
                                                  angleDeflection=angleDeflection)
        fresnelIntersectionFactory = mock(FresnelIntersectionFactory)
        when(fresnelIntersectionFactory).compute(...).thenReturn(fresnelIntersection)
        return fresnelIntersectionFactory

    @staticmethod
    def _createLogger():
        logger = mock(Logger)
        when(logger).logPoint(...).thenReturn()
        when(logger).logDataPoint(...).thenReturn()
        return logger
