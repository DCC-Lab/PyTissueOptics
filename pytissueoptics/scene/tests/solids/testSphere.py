import unittest
import math

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Sphere


class TestSphere(unittest.TestCase):
    def testGivenANewDefaultSphere_shouldBePlacedAtOrigin(self):
        sphere = Sphere()
        self.assertEqual(Vector(0, 0, 0), sphere.position)

    def testGivenANewSphere_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        sphere = Sphere(position=position)
        self.assertEqual(Vector(2, 2, 1), sphere.position)

    def testGivenANewDefaultSphere_shouldHaveARadiusOf1(self):
        sphere = Sphere()
        self.assertEqual(1, sphere.radius)

    def testGivenASphere_shouldHaveCorrectVerticesLength(self):
        sphere = Sphere(radius=2)
        self.assertEqual(2, sphere._vertices[0].getNorm())

    def testGivenALowOrderSphere_shouldNotApproachCorrectSphereArea(self):
        sphere = Sphere()
        icosphereArea = 0
        perfectSphereArea = 4 * math.pi * sphere.radius ** 2

        for polygon in sphere.getPolygons():
            icosphereArea += 0.5 * polygon.vertices[0].cross(polygon.vertices[1]).getNorm()

        self.assertNotAlmostEqual(perfectSphereArea, icosphereArea, 3)

    def testGivenAHighOrderSphere_shouldApproachCorrectSphereArea(self):
        sphere = Sphere(radius=1, order=5)
        icosphereArea = 0
        perfectSphereArea = 4 * math.pi * sphere.radius ** 2

        for polygon in sphere.getPolygons():
            AB = polygon.vertices[0] - polygon.vertices[1]
            AC = polygon.vertices[0] - polygon.vertices[2]
            icosphereArea += 0.5 * AB.cross(AC).getNorm()

        self.assertAlmostEqual(perfectSphereArea, icosphereArea, 2)
