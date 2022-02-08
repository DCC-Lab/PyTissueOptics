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

    def testGivenANewDefault_shouldHaveARadiusOf1(self):
        ellipsoid = Ellipsoid()
        self.assertEqual(1, ellipsoid.radius)

    def testGivenALowOrderEllipsoid_shouldNotApproachCorrectEllipsoidArea(self):
        ellipsoid = Ellipsoid()
        p = 8/5
        a = ellipsoid._a
        b = ellipsoid._b
        c = ellipsoid._c
        ellipsoidArea = 0
        perfectEllipsoidArea = 4*math.pi*((a**p*b**p + a**p*c**p + b**p*c**p)/3)**(1/p)

        for surface in ellipsoid._surfaces["noLabel"]:
            ellipsoidArea += 0.5 * surface.vertices[0].cross(surface.vertices[1]).getNorm()
        print(perfectEllipsoidArea, ellipsoidArea)
        self.assertNotAlmostEqual(perfectEllipsoidArea, ellipsoidArea, 1)

    def testGivenAHighOrderEllipsoid_shouldApproachCorrectEllipsoidArea(self):
        ellipsoid = Ellipsoid(a=2, b=3, c=1, order=5)
        p = 8 / 5
        a = ellipsoid._a
        b = ellipsoid._b
        c = ellipsoid._c
        ellipsoidArea = 0
        perfectEllipsoidArea = 4 * math.pi * ((a ** p * b ** p + a ** p * c ** p + b ** p * c ** p) / 3) ** (1 / p)

        for surface in ellipsoid._surfaces["noLabel"]:
            AB = surface.vertices[0]-surface.vertices[1]
            AC = surface.vertices[0]-surface.vertices[2]
            ellipsoidArea += 0.5 * AB.cross(AC).getNorm()

        self.assertAlmostEqual(perfectEllipsoidArea, ellipsoidArea, 2)
