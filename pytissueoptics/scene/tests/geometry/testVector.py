import unittest

import numpy as np

from pytissueoptics.scene.geometry import Vector


class TestVector(unittest.TestCase):

    def setUp(self):
        self.vector = Vector(x=1, y=2.4, z=0.5)

    def testGivenAnyVectorNotNormalized_shouldNotHaveANormOf1(self):
        self.assertNotEqual(1, self.vector.getNorm())

    def testWhenNormalize_shouldHaveANormOf1(self):
        self.vector.normalize()
        self.assertEqual(1, self.vector.getNorm())

    def testGivenAZeroVector_whenNormalize_shouldStayAZeroVector(self):
        zeroVector = Vector(0, 0, 0)
        zeroVector.normalize()
        self.assertEqual(0, zeroVector.getNorm())

    def testWhenAddVector_shouldAddTheOtherVectorToTheOriginalVector(self):
        initialNorm = self.vector.getNorm()
        anotherVector = Vector(1, 1, 1)

        self.vector.add(anotherVector)

        self.assertNotEqual(initialNorm, self.vector.getNorm())

    def testWhenSubtractVector_shouldSubtractTheOtherVectorToTheOriginalVector(self):
        initialNorm = self.vector.getNorm()
        anotherVector = Vector(1, 1, 1)

        self.vector.subtract(anotherVector)

        self.assertNotEqual(initialNorm, self.vector.getNorm())

    def testWhenMultiplyWithScalar_shouldMultiplyAllItsComponents(self):
        initialNorm = self.vector.getNorm()
        multiplier = 2.5

        self.vector.multiply(multiplier)

        self.assertEqual(initialNorm * multiplier, self.vector.getNorm())

    def testWhenDivideWithScalar_shouldDivideAllItsComponents(self):
        initialNorm = self.vector.getNorm()
        divider = 2.5

        self.vector.divide(divider)

        self.assertEqual(initialNorm / divider, self.vector.getNorm())

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

    def testWhenDividingWithDivisionOperator_shouldCreateANewVector(self):
        divider = 2
        initialNorm = self.vector.getNorm()

        newVector = self.vector / divider

        self.assertEqual(initialNorm / divider, newVector.getNorm())

    def testWhenMultiplyingWithMultiplicationOperator_shouldCreateANewVector(self):
        multiplier = 2
        initialNorm = self.vector.getNorm()

        newVector = self.vector * multiplier

        self.assertEqual(initialNorm * multiplier, newVector.getNorm())

    def testGivenTwoVectorsWithTheSameCoordinates_shouldBeEqual(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(-1, 0, 1)
        self.assertTrue(vector == vector2)

    def testGivenTwoDifferentVectors_shouldNotBeEqual(self):
        vector = Vector(-1, 0, 1)
        vector2 = Vector(1, 0, 1)
        self.assertFalse(vector == vector2)

    def testWhenUpdate_shouldChangeTheVectorComponents(self):
        self.vector.update(0, 0, 0)
        self.assertEqual(Vector(0, 0, 0), self.vector)

    def testWhenCopy_shouldGiveAWithTheSameComponents(self):
        newVector = self.vector.copy()
        self.assertEqual(newVector, self.vector)

    def testWhenCopy_shouldGiveANewVector(self):
        newVector = self.vector.copy()
        newVector.add(Vector(1, 1, 1))
        self.assertNotEqual(newVector, self.vector)

    def testWhenCheckIfAnotherVectorWithTheSameCoordinatesIsContained_shouldReturnTrue(self):
        vector = Vector(0, 0, 0)
        listOfOtherVectors = [Vector(0, 0, 0)]
        self.assertTrue(vector in listOfOtherVectors)

    def testWhenGetAnyOrthogonal_shouldReturnANewOrthogonalVector(self):
        N = 1000
        direction_xs = np.random.uniform(-1, 1, N)
        direction_ys = np.random.uniform(-1, 1, N)
        direction_zs = np.random.uniform(-1, 1, N)
        for i in range(N):
            vector = Vector(direction_xs[i], direction_ys[i], direction_zs[i])
            orthogonalVector = vector.getAnyOrthogonal()
            self.assertEqual(0, vector.dot(orthogonalVector))
            self.assertNotEqual(0, orthogonalVector.getNorm())

        edgeCaseVectors = [Vector(-1, 0, 0), Vector(1, 0, 0),
                           Vector(0, 1, 0), Vector(0, -1, 0),
                           Vector(0, 0, 1), Vector(0, 0, -1)]
        for vector in edgeCaseVectors:
            orthogonalVector = vector.getAnyOrthogonal()
            self.assertEqual(0, vector.dot(orthogonalVector))
            self.assertNotEqual(0, orthogonalVector.getNorm())

    def testGivenZeroVector_whenGetAnyOrthogonal_shouldReturnZeroVector(self):
        zeroVector = Vector(0, 0, 0)
        orthogonalVector = zeroVector.getAnyOrthogonal()
        self.assertEqual(0, orthogonalVector.getNorm())

    def testGivenTwoDifferentVectors_shouldHaveDifferentHashes(self):
        vector1 = Vector(0, 0, 0)
        vector2 = Vector(1, 0, 0)
        self.assertNotEqual(hash(vector1), hash(vector2))

    def testGivenTwoVectorsWithTheSameCoordinates_shouldHaveTheSameHash(self):
        vector1 = Vector(0, 0, 1)
        vector2 = Vector(0, 0, 1)
        self.assertEqual(hash(vector1), hash(vector2))

    def testGivenVectorTowardsXAxis_whenRotateAroundZAxisBy90Degrees_shouldPointTowardsYAxis(self):
        vector = Vector(1, 0, 0)
        vector.rotateAround(Vector(0, 0, 1), np.pi / 2)
        self.assertEqual(Vector(0, 1, 0), vector)

    def testWhenPrintVector_shouldPrintComponents(self):
        vector = Vector(1, 2, 3)
        self.assertEqual("<Vector>:(1, 2, 3)", str(vector))
