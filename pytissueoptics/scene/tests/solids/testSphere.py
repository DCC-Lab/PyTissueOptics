import math
import unittest

from pytissueoptics.scene.geometry import Vector, Vertex, primitives
from pytissueoptics.scene.solids import Sphere


class TestSphere(unittest.TestCase):
    def testGivenANewDefaultSphere_shouldBePlacedAtOrigin(self):
        sphere = Sphere()
        self.assertEqual(Vector(0, 0, 0), sphere.position)

    def testGivenANewSphere_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        sphere = Sphere(position=position)
        self.assertEqual(Vector(2, 2, 1), sphere.position)

    def testGivenANewSphereWithQuadPrimitive_shouldNotCreateSphere(self):
        with self.assertRaises(NotImplementedError):
            Sphere(primitive=primitives.QUAD)

    def testGivenANewDefaultSphere_shouldHaveARadiusOf1(self):
        sphere = Sphere()
        self.assertEqual(1, sphere.radius)

    def testGivenASphere_shouldHaveCorrectVerticesLength(self):
        sphere = Sphere(radius=2)
        self.assertEqual(2, sphere._vertices[0].getNorm())

    def testGivenALowOrderSphere_shouldNotApproachCorrectSphereArea(self):
        sphere = Sphere()
        icosphereArea = 0
        perfectSphereArea = 4 * math.pi * sphere.radius**2

        for polygon in sphere.getPolygons():
            icosphereArea += 0.5 * polygon.vertices[0].cross(polygon.vertices[1]).getNorm()

        self.assertNotAlmostEqual(perfectSphereArea, icosphereArea, 3)

    def testGivenAHighOrderSphere_shouldApproachCorrectSphereArea(self):
        sphere = Sphere(radius=1, order=4)
        icosphereArea = 0
        perfectSphereArea = 4 * math.pi * sphere.radius**2
        tolerance = 0.002

        for polygon in sphere.getPolygons():
            AB = polygon.vertices[0] - polygon.vertices[1]
            AC = polygon.vertices[0] - polygon.vertices[2]
            icosphereArea += 0.5 * AB.cross(AC).getNorm()

        self.assertAlmostEqual(perfectSphereArea, icosphereArea, delta=tolerance * perfectSphereArea)

    def testWhenContainsWithVerticesThatAreAllInsideTheSphere_shouldReturnTrue(self):
        sphere = Sphere(1, position=Vector(2, 2, 0))
        vertices = [Vertex(2.5, 2.5, 0), Vertex(2, 2, 0)]

        self.assertTrue(sphere.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotAllInsideTheSphere_shouldReturnFalse(self):
        sphere = Sphere(1, position=Vector(2, 2, 0))
        vertices = [Vertex(3, 3, 1), Vertex(2, 2, 0)]

        self.assertFalse(sphere.contains(*vertices))
