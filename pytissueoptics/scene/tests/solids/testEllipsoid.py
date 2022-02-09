import unittest
import math

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Ellipsoid


class TestSphere(unittest.TestCase):
    def testGivenANewDefault_shouldBePlacedAtOrigin(self):
        ellipsoid = Ellipsoid()
        self.assertEqual(Vector(0, 0, 0), ellipsoid.position)

    def testGivenANew_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        ellipsoid = Ellipsoid(position=position)
        self.assertEqual(Vector(2, 2, 1), ellipsoid.position)

    def testGivenANewDefault_shouldHaveARadiusOfNone(self):
        ellipsoid = Ellipsoid()
        self.assertIsNone(ellipsoid.radius)

    def testGivenALowOrderSphericalEllipsoid_shouldApproachCorrectSphereAreaTo5Percent(self):
        ellipsoid = Ellipsoid(a=1, b=1, c=1, order=3)
        perfectSphereArea = 4*math.pi
        tolerance = 0.05

        ellipsoidArea = self._getTotalTrianglesArea(ellipsoid._surfaceDict["noLabel"])

        self.assertAlmostEqual(perfectSphereArea, ellipsoidArea, delta=tolerance*perfectSphereArea)

    def testGivenALowOrderEllipsoid_shouldApproachCorrectEllipsoidAreaTo5Percent(self):
        ellipsoid = Ellipsoid(a=2, b=3, c=5, order=3)
        p = 8/5
        a = ellipsoid._a
        b = ellipsoid._b
        c = ellipsoid._c
        tolerance = 0.05

        perfectEllipsoidArea = 4*math.pi*((a**p*b**p + a**p*c**p + b**p*c**p)/3)**(1/p)
        ellipsoidArea = self._getTotalTrianglesArea(ellipsoid._surfaceDict["noLabel"])

        self.assertAlmostEqual(perfectEllipsoidArea, ellipsoidArea, delta=perfectEllipsoidArea*tolerance)
        self.assertNotAlmostEqual(perfectEllipsoidArea, ellipsoidArea, 1)

    def testGivenAHighOrderEllipsoid_shouldApproachCorrectEllipsoidAreaTo1Percent(self):
        ellipsoid = Ellipsoid(a=2, b=3, c=5, order=6)
        p = 8 / 5
        a = ellipsoid._a
        b = ellipsoid._b
        c = ellipsoid._c
        tolerance = 0.01

        perfectEllipsoidArea = 4 * math.pi * ((a ** p * b ** p + a ** p * c ** p + b ** p * c ** p) / 3) ** (1 / p)
        ellipsoidArea = self._getTotalTrianglesArea(ellipsoid._surfaceDict["noLabel"])

        self.assertAlmostEqual(perfectEllipsoidArea, ellipsoidArea, delta=tolerance*perfectEllipsoidArea)

    @staticmethod
    def _getTotalTrianglesArea(surfaces):
        totalArea = 0
        for surface in surfaces:
            AB = surface.vertices[0] - surface.vertices[1]
            AC = surface.vertices[0] - surface.vertices[2]
            totalArea += 0.5 * AB.cross(AC).getNorm()
        return totalArea
