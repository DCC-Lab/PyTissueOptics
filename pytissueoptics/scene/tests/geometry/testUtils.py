from dataclasses import dataclass
import unittest

import numpy as np

from pytissueoptics.scene.geometry import Rotation
from pytissueoptics.scene.geometry.utils import eulerRotationMatrix, rotateVerticesArray, getAxisAngleBetween
from pytissueoptics.scene.geometry.vector import Vector


class TestEulerRotationMatrix(unittest.TestCase):
    def testGivenNoRotation_shouldNotRotate(self):
        noRotation = eulerRotationMatrix()
        p = [1, 1, 1]

        pRotated = np.dot(noRotation, p)

        self.assertTrue(np.allclose(p, pRotated))

    def testGivenXRotation_shouldRotateAroundX(self):
        xRotation = eulerRotationMatrix(xTheta=90)
        p = [1, 1, 1]

        pRotated = np.dot(xRotation, p)

        self.assertTrue(np.allclose([1, -1, 1], pRotated))

    def testGivenYRotation_shouldRotateAroundY(self):
        yRotation = eulerRotationMatrix(yTheta=90)
        p = [1, 1, 1]

        pRotated = np.dot(yRotation, p)

        self.assertTrue(np.allclose([1, 1, -1], pRotated))

    def testGivenZRotation_shouldRotateAroundZ(self):
        zRotation = eulerRotationMatrix(zTheta=90)
        p = [1, 1, 1]

        pRotated = np.dot(zRotation, p)

        self.assertTrue(np.allclose([-1, 1, 1], pRotated))

    def testGivenXYRotation_shouldRotateAroundXYInOrder(self):
        rotation = eulerRotationMatrix(xTheta=90, yTheta=90)
        p = [1, 1, 1]

        pRotated = np.dot(rotation, p)

        self.assertTrue(np.allclose([1, -1, -1], pRotated))

    def testGivenXYZRotation_shouldRotateAroundXYZInOrder(self):
        rotation = eulerRotationMatrix(xTheta=90, yTheta=90, zTheta=90)
        p = [1, 1, 1]

        pRotated = np.dot(rotation, p)

        self.assertTrue(np.allclose([1, 1, -1], pRotated))

    def testGivenInverseRotation_shouldInverseRotation(self):
        rotation = eulerRotationMatrix(xTheta=90, yTheta=90, zTheta=90)
        inverseRotation = eulerRotationMatrix(xTheta=90, yTheta=90, zTheta=90, inverse=True)
        p = [1, 1, 1]

        pRotated = np.dot(rotation, p)
        pUnrotated = np.dot(inverseRotation, pRotated)

        self.assertTrue(np.allclose([1, 1, 1], pUnrotated))


class TestRotateVerticesArray(unittest.TestCase):
    def testShouldRotateAllVertices(self):
        p0, p1 = [1, 1, 1], [-1, -1, -1]
        verticesArray = np.asarray([p0, p1])
        rotation = Rotation(90, 90, 90)
        rotatedVerticesArray = rotateVerticesArray(verticesArray, rotation)

        p0Rotated, p1Rotated = rotatedVerticesArray

        self.assertTrue(np.allclose([1, 1, -1], p0Rotated))
        self.assertTrue(np.allclose([-1, -1, 1], p1Rotated))


@dataclass
class AxisAngleTestCase:
    v1: Vector
    v2: Vector
    expectedAxis: Vector
    expectedAngle: float


class TestGetAxisAngleBetween(unittest.TestCase):
    def testGivenTwoVectorsOnABaseAxisPlaneWithLessThan180DegreesBetween_shouldReturnAxisAndAngleOfRotation(self):
        testCases = self._generateAxisAlignedTestCases()
        for testCase in testCases:
            axis, angle = getAxisAngleBetween(testCase.v1, testCase.v2)
            self.assertTrue(np.allclose(testCase.expectedAxis.array, axis.array))
            self.assertTrue(np.isclose(testCase.expectedAngle, angle))

            axis, angle = getAxisAngleBetween(testCase.v2, testCase.v1)
            testCase.expectedAxis *= -1
            self.assertTrue(np.allclose(testCase.expectedAxis.array, axis.array))
            self.assertTrue(np.isclose(testCase.expectedAngle, angle))

    def _generateAxisAlignedTestCases(self):
        testCases = []
        for axis in [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]:
            for (rotationAxis, rotationAxisKey) in zip([Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)], ['xTheta', 'yTheta', 'zTheta']):
                if axis == rotationAxis:
                    continue
                for angle in [-180, -150, -70, -10, 10, 70, 150, 180]:
                    v1 = axis
                    v2 = rotateVerticesArray([v1.array], Rotation(**{rotationAxisKey: angle}))[0]
                    v2 = Vector(*v2)
                    rotationAxisCopy = rotationAxis.copy()
                    if angle < 0:
                        rotationAxisCopy *= -1
                        angle = -angle
                    testCases.append(AxisAngleTestCase(v1, v2, rotationAxisCopy, np.radians(angle)))
        return testCases

    def testGivenTwoVectorsWithSameOrientation_shouldReturnZeroAngle(self):
        v1 = Vector(1, 0, 0)
        v2 = Vector(1, 0, 0)
        _, angle = getAxisAngleBetween(v1, v2)
        self.assertTrue(np.isclose(0, angle))
