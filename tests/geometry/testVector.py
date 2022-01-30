import unittest
from python_graphics_engine.geometry import Vector


class TestVector(unittest.TestCase):

    def setUp(self):
        self.vector = Vector(x=1, y=2.4, z=0.5)

    def testWhenAskingNorm_shouldReturnCorrectNorm(self):
        vector = Vector(1, 1, 1)
        self.assertEqual(3**(1/2), vector.norm())

    def testWhenNormalize_shouldHaveNormOf1(self):
        self.vector.normalize()
        self.assertEqual(1, self.vector.norm())

    def testWhenAddingVectors_shouldCreateANewVector(self):
        initialNorm = self.vector.norm()
        anotherVector = Vector(1, 1, 1)

        newVector = self.vector + anotherVector

        self.assertEqual(initialNorm, self.vector.norm())
        self.assertEqual(self.vector.x + anotherVector.x, newVector.x)

    def testWhenSubtractingVectors_shouldCreateANewVector(self):
        initialNorm = self.vector.norm()
        anotherVector = Vector(1, 1, 1)

        newVector = self.vector + anotherVector

        self.assertEqual(initialNorm, self.vector.norm())
        self.assertEqual(self.vector.x + anotherVector.x, newVector.x)

    def testGivenSameVector_whenEquals_shouldBeEqual(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(-1, 0, 1)
        self.assertTrue(vector == vector2)

    def testGivenDifferentVector_whenEquals_shouldNotBeEqual(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(1, 0, 1)
        self.assertFalse(vector == vector2)
