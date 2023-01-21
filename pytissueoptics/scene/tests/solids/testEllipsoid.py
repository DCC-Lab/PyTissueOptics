import unittest
import math

from pytissueoptics.scene.geometry import Vector, Vertex, primitives
from pytissueoptics.scene.solids import Ellipsoid


class TestEllipsoid(unittest.TestCase):
    def testGivenNewEllipsoidWithNoPosition_shouldBePlacedAtOrigin(self):
        ellipsoid = Ellipsoid()
        self.assertEqual(Vector(0, 0, 0), ellipsoid.position)

    def testGivenNewEllipsoidWithPosition_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        ellipsoid = Ellipsoid(position=position)
        self.assertEqual(Vector(2, 2, 1), ellipsoid.position)

    def testGivenNewEllipsoidWithQuadPrimitive_shouldNotCreateEllipsoid(self):
        with self.assertRaises(NotImplementedError):
            Ellipsoid(primitive=primitives.QUAD)

    def testGivenALowOrderSphericalEllipsoid_shouldApproachCorrectSphereAreaTo5Percent(self):
        ellipsoid = Ellipsoid(a=1, b=1, c=1, order=3)
        perfectSphereArea = 4 * math.pi
        tolerance = 0.05
        ellipsoidArea = self._getTotalTrianglesArea(ellipsoid.getPolygons())

        self.assertAlmostEqual(perfectSphereArea, ellipsoidArea, delta=tolerance * perfectSphereArea)

    def testGivenALowOrderEllipsoid_shouldApproachCorrectEllipsoidAreaTo5Percent(self):
        ellipsoid = Ellipsoid(a=2, b=3, c=5, order=2)
        perfectEllipsoidArea = self._getPerfectEllipsoidArea(ellipsoid)
        tolerance = 0.05
        ellipsoidArea = self._getTotalTrianglesArea(ellipsoid.getPolygons())

        self.assertAlmostEqual(perfectEllipsoidArea, ellipsoidArea, delta=perfectEllipsoidArea * tolerance)
        self.assertNotAlmostEqual(perfectEllipsoidArea, ellipsoidArea, 1)

    def testGivenAHighOrderEllipsoid_shouldApproachCorrectEllipsoidAreaTo1Percent(self):
        ellipsoid = Ellipsoid(a=2, b=3, c=5, order=3)
        perfectEllipsoidArea = self._getPerfectEllipsoidArea(ellipsoid)
        tolerance = 0.01
        ellipsoidArea = self._getTotalTrianglesArea(ellipsoid.getPolygons())

        self.assertAlmostEqual(perfectEllipsoidArea, ellipsoidArea, delta=tolerance * perfectEllipsoidArea)

    @staticmethod
    def _getTotalTrianglesArea(surfaces):
        totalArea = 0
        for surface in surfaces:
            AB = surface.vertices[0] - surface.vertices[1]
            AC = surface.vertices[0] - surface.vertices[2]
            totalArea += 0.5 * AB.cross(AC).getNorm()
        return totalArea

    @staticmethod
    def _getPerfectEllipsoidArea(ellipsoid: Ellipsoid) -> float:
        p = 8 / 5
        a = ellipsoid._a
        b = ellipsoid._b
        c = ellipsoid._c
        return 4 * math.pi * ((a ** p * b ** p + a ** p * c ** p + b ** p * c ** p) / 3) ** (1 / p)

    def testWhenContainsWithVerticesThatAreAllInsideTheEllipsoid_shouldReturnTrue(self):
        ellipsoid = Ellipsoid(3, 1, 1, position=Vector(2, 2, 0))
        ellipsoid.rotate(0, 0, 30)
        vertices = [Vertex(3.3, 3, 0), Vertex(2, 2, 0)]

        self.assertTrue(ellipsoid.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotAllInsideTheEllipsoid_shouldReturnFalse(self):
        ellipsoid = Ellipsoid(3, 1, 1, position=Vector(2, 2, 0))
        vertices = [Vertex(3.4, 2.9, 0), Vertex(2, 2, 0)]

        self.assertFalse(ellipsoid.contains(*vertices))
