import unittest

from graphics.geometry import Vector


class TestVector(unittest.TestCase):

    def setUp(self):
        self.vector = Vector(x=1, y=2.4, z=0.5)

    def testWhenNormalize_shouldHaveNormOf1(self):
        self.vector.normalize()
        self.assertEqual(1, self.vector.norm)

    def testWhenAddingVectors_shouldCreateANewVector(self):
        initialNorm = self.vector.norm
        anotherVector = Vector(1, 1, 1)

        newVector = self.vector + anotherVector

        self.assertEqual(initialNorm, self.vector.norm)
        self.assertEqual(self.vector.x + anotherVector.x, newVector.x)
