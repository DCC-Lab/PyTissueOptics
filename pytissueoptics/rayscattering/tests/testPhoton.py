import math
import unittest

from pytissueoptics.rayscattering import Photon
from pytissueoptics.scene import Vector, Material
from pytissueoptics.scene.geometry import Polygon
from pytissueoptics.scene.intersection.intersectionFinder import Intersection


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
        # fixme: should allow for no intersectionFinder ?
        pass

    def testWhenMoveBy_shouldMovePhotonByTheGivenDistanceTowardsItsDirection(self):
        initialPosition = self.POSITION.copy()
        self.photon.moveBy(2)
        self.assertEqual(initialPosition + self.DIRECTION * 2, self.photon.position)

    def testWhenRefractWithPerpendicularSurface_shouldNotRotatePhoton(self):
        initialDirection = self.DIRECTION.copy()
        n1, n2 = 1, 1.4
        surfaceElement = Polygon([Vector()], normal=Vector(0, 0, 1),
                                 insideMaterial=Material(index=n2),
                                 outsideMaterial=Material(index=n1))
        intersection = Intersection(1, self.POSITION + self.DIRECTION, surfaceElement)

        self.photon.refract(intersection)

        self.assertEqual(initialDirection, self.photon.direction)

    def testWhenRefract_shouldRotatePhotonTowardsFresnelRefractionAngle(self):
        # todo: move logic to FresnelIntersect(intersection)
        incidenceAngle = math.pi / 10
        self.photon = Photon(self.POSITION, Vector(0, math.sin(incidenceAngle), -math.cos(incidenceAngle)))
        n1, n2 = 1, 1.4
        surfaceElement = Polygon([Vector()], normal=Vector(0, 0, 1),
                                 insideMaterial=Material(index=n2),
                                 outsideMaterial=Material(index=n1))
        intersection = Intersection(1, self.POSITION + self.DIRECTION, surfaceElement)

        self.photon.refract(intersection)

        refractionAngle = math.asin(n1/n2 * math.sin(incidenceAngle))
        expectedDirection = Vector(0, math.sin(refractionAngle), -math.cos(refractionAngle))
        self.assertVectorEquals(expectedDirection, self.photon.direction)

    def testGivenALogger_whenPropagate_shouldLogInitialPosition(self):
        pass

    def testGivenALogger_whenPropagate_shouldLogIntersectionPositions(self):
        pass

    def testGivenALogger_whenScatter_shouldLogWeightLossAtThisPosition(self):
        pass

    def assertVectorEquals(self, v1, v2):
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertAlmostEqual(v1.z, v2.z)
