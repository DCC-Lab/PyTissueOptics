import unittest
from python_graphic_engine.geometry import *


class TestVector(unittest.TestCase):

    def test_withEmpty_onCreate_shouldNotBeNone(self):
        vector = Vector()
        self.assertIsNotNone(vector)

    def test_withXYZ_onXYZPropertyCall_shouldBeXYZ(self, vector=None, values=[]):
        if vector is not None:
            vector = vector
            values = values
        else:
            vector = Vector(1, -2, 0)
            values = [1, -2, 0]

        with self.subTest("x"):
            self.assertEqual(vector.x, values[0])

        with self.subTest("y"):
            self.assertEqual(vector.y, values[1])

        with self.subTest("z"):
            self.assertEqual(vector.z, values[2])

    def test_withSameVector_onEqCondition_shouldReturnTrue(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(-1, 0, 1)
        self.assertTrue(vector == vector2)

    def test_withDifferentVector_onEqCondition_shouldReturnFalse(self):
        vector = Vector(1, 0, -1)
        vector2 = Vector(-1, 0, 1)
        self.assertFalse(vector == vector2)

    def test_withAlmostSameVector_E9_onEqCondition_shouldReturnFalse(self):
        epsilon = 1e-9
        vector = Vector(-1 + epsilon, 0, 1)
        vector2 = Vector(-1, 0, 1)
        self.assertFalse(vector == vector2)

    def test_withVector_onAdd_shouldReturnVector(self):
        vector = Vector(1, 2, 3)
        vector2 = Vector(-1, 0, 1)
        vector3 = vector + vector2
        self.assertTrue(isinstance(vector3, Vector))

    def test_withVector_onAdd_shouldReturnAddedVectors(self):
        vector = Vector(1, 2, 3)
        vector2 = Vector(-1, 0, 1)
        vector3 = vector + vector2
        assertVector = Vector(0, 2, 4)
        self.assertEqual(assertVector, vector3)

    def test_withVector_onSubtraction_shouldReturnSubtractedVectors(self):
        vector = Vector(1, 2, 3)
        vector2 = Vector(-1, 0, 1)
        vector3 = vector - vector2
        assertVector = Vector(2, 2, 2)
        self.assertEqual(assertVector, vector3)

    def test_givenNullValue_onNorm_shouldReturn0(self):
        vector = Vector(0, 0, 0)
        self.assertEqual(0, vector.norm())

    def test_givenNegativeValue_onNorm_shouldReturnCorrectNorm(self):
        vector = Vector(-1, -3, 0)
        self.assertEqual(10 ** (1 / 2), vector.norm())


if __name__ == "__main__":
    unittest.main()
