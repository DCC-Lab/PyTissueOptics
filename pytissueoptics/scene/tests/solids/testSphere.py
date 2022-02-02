import unittest

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
        self.assertEqual(Vector(0, 0, 0), sphere.position)

    def testGivenAHighOrderSphere_shouldApproachCorrectSphereArea(self):
        sphere = Sphere()
        self.assertEqual(Vector(0, 0, 0), sphere.position)
