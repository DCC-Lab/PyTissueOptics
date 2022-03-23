import math
import unittest

from mockito import mock, when

from pytissueoptics.rayscattering import Photon
from pytissueoptics.rayscattering.fresnel import FresnelIntersection, FresnelIntersectionFactory
from pytissueoptics.scene import Vector, Material
from pytissueoptics.scene.geometry import Polygon
from pytissueoptics.scene.intersection.intersectionFinder import Intersection, IntersectionFinder


class TestPhoton(unittest.TestCase):
    def setUp(self):
        self.POSITION = Vector(2, 2, 0)
        self.DIRECTION = Vector(0, 0, -1)
        self.photon = Photon(self.POSITION, self.DIRECTION)

    def testShouldBeInTheGivenState(self):
        self.assertEqual(self.POSITION, self.photon.position)
        self.assertEqual(self.DIRECTION, self.photon.direction)

    def testShouldBeAlive(self):
        self.assertTrue(self.photon.isAlive)

    def testGivenNoContext_whenPropagate_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.photon.propagate()

    def testWhenSetContext_shouldSetCurrentMaterialAsWorldMaterial(self):
        # fixme: should set proper initial material
        worldMaterial = Material()
        self.photon.setContext(worldMaterial)
        self.assertEqual(worldMaterial, self.photon.material)

    def testWhenMoveBy_shouldMovePhotonByTheGivenDistanceTowardsItsDirection(self):
        initialPosition = self.POSITION.copy()
        self.photon.moveBy(2)
        self.assertEqual(initialPosition + self.DIRECTION * 2, self.photon.position)

    def testWhenRefract_shouldOrientPhotonTowardsFresnelRefractionAngle(self):
        incidenceAngle = math.pi / 10
        self.photon = Photon(self.POSITION, Vector(0, math.sin(incidenceAngle), -math.cos(incidenceAngle)))
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

        self.assertEqual(1 - material.albedo, self.photon.weight)

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
        self.photon = Photon(self.POSITION, Vector(0, math.sin(incidenceAngle), -math.cos(incidenceAngle)))
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

    def createIntersectionFinder(self, intersectionDistance, rayLength):
        aPolygon = Polygon([Vector(), Vector(), Vector()],
                           insideMaterial=Material(), outsideMaterial=Material())
        intersection = Intersection(intersectionDistance, position=Vector(), polygon=aPolygon,
                                    distanceLeft=rayLength-intersectionDistance)
        intersectionFinder = mock(IntersectionFinder)
        when(intersectionFinder).findIntersection(...).thenReturn(intersection)
        return intersectionFinder

    def testWhenStepWithIntersection_shouldMovePhotonToIntersection(self):
        distance = 8
        intersectionFinder = self.createIntersectionFinder(distance, rayLength=distance+2)
        initialPosition = self.POSITION.copy()
        self.photon.setContext(Material(), intersectionFinder=intersectionFinder)

        self.photon.step(distance + 2)

        self.assertVectorEqual(initialPosition + self.photon.direction * distance,
                               self.photon.position)

    def testWhenStepWithNoIntersection_shouldMovePhotonAcrossStepDistanceAndScatter(self):
        self.fail()

    def testWhenStepWithNoIntersection_shouldReturnADistanceLeftOfZero(self):
        self.fail()

    def testWhenStepWithReflectingIntersection_shouldReflect(self):
        self.fail()

    def testWhenStepWithReflectingIntersection_shouldReturnDistanceLeft(self):
        totalDistance = 10
        intersectionDistance = 8
        intersectionFinder = self.createIntersectionFinder(intersectionDistance, totalDistance)
        material = mock(Material)
        when(material).getScatteringDistance().thenReturn(totalDistance)

        fresnelIntersection = FresnelIntersection(Material(), Vector(),
                                                  isReflected=True, angleDeflection=0.1)
        fresnelIntersectionFactory = mock(FresnelIntersectionFactory)
        when(fresnelIntersectionFactory).compute(...).thenReturn(fresnelIntersection)
        self.photon.setContext(material, intersectionFinder=intersectionFinder,
                               fresnelIntersectionFactory=fresnelIntersectionFactory)

        distanceLeft = self.photon.step(totalDistance)

        self.assertEqual(totalDistance - intersectionDistance, distanceLeft)

    def testWhenStepWithRefractingIntersection_shouldRefract(self):
        self.fail()

    def testWhenStepWithRefractingIntersection_shouldUpdatePhotonMaterialToNextMaterial(self):
        self.fail()

    def testWhenStepWithRefractingIntersection_shouldReturnDistanceLeftInNextMaterial(self):
        self.fail()

    # todo: find some test cases for the moveBy(1e-3) after reflect/refract

    def testGivenALogger_whenPropagate_shouldLogInitialPosition(self):
        pass

    def testGivenALogger_whenPropagate_shouldLogIntersectionPositions(self):
        pass

    def testGivenALogger_whenScatter_shouldLogWeightLossAtThisPosition(self):
        pass

    def assertVectorEqual(self, v1, v2):
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertAlmostEqual(v1.z, v2.z)

    def assertVectorNotEqual(self, v1, v2):
        self.assertNotAlmostEqual(v1.x, v2.x)
        self.assertNotAlmostEqual(v1.y, v2.y)
        self.assertNotAlmostEqual(v1.z, v2.z)
