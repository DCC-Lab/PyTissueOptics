import unittest
from python_graphics_engine.geometry import Vector


class TestVector(unittest.TestCase):

    def setUp(self):
        self.vector = Vector(x=1, y=2.4, z=0.5)

    def testGivenAnyVectorNotNormalized_shouldNotHaveANormOf1(self):
        self.assertNotEqual(1, self.vector.getNorm())

    def testWhenNormalize_shouldHaveANormOf1(self):
        self.vector.normalize()
        self.assertEqual(1, self.vector.getNorm())

    def testWhenAddingVectorsWithAdditionOperator_shouldCreateANewVector(self):
        initialNorm = self.vector.getNorm()
        anotherVector = Vector(1, 1, 1)

        newVector = self.vector + anotherVector

        self.assertEqual(initialNorm, self.vector.getNorm())
        self.assertEqual(self.vector.x + anotherVector.x, newVector.x)

    def testWhenSubtractingVectorsWithSubtractionOperator_shouldCreateANewVector(self):
        initialNorm = self.vector.getNorm()
        anotherVector = Vector(1, 1, 1)

        newVector = self.vector - anotherVector

        self.assertEqual(initialNorm, self.vector.getNorm())
        self.assertEqual(self.vector.x - anotherVector.x, newVector.x)

    def testGivenTwoVectorsWithTheSameCoordinates_shouldBeEqual(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(-1, 0, 1)
        self.assertTrue(vector == vector2)

    def testGivenTwoDifferentVectors_shouldNotBeEqual(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(1, 0, 1)
        self.assertFalse(vector == vector2)
