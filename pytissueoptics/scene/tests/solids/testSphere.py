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
        perfectSphereArea = 4*math.pi*sphere.radius**2

        for surface in sphere._surfaces["Sphere"]:
            icosphereArea += 0.5 * surface.vertices[0].cross(surface.vertices[1]).getNorm()
        print(icosphereArea, perfectSphereArea)

        self.assertNotAlmostEqual(perfectSphereArea, icosphereArea, 3)

    def testGivenAHighOrderSphere_shouldApproachCorrectSphereArea(self):
        sphere = Sphere(radius=1, order=5)
        icosphereArea = 0
        perfectSphereArea = 4 * math.pi * sphere.radius ** 2

        for surface in sphere._surfaces["Sphere"]:
            AB = surface.vertices[0]-surface.vertices[1]
            AC = surface.vertices[0]-surface.vertices[2]
            icosphereArea += 0.5 * AB.cross(AC).getNorm()

        print(icosphereArea, perfectSphereArea)

        self.assertAlmostEqual(perfectSphereArea, icosphereArea, 2)
