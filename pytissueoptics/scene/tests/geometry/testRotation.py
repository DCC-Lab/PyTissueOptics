import unittest

from pytissueoptics.scene.geometry import Rotation
from pytissueoptics.scene.geometry.vector import Vector


class TestRotation(unittest.TestCase):
    def testGivenDefaultRotation_shouldBeAlignedWithAxes(self):
        rotation = Rotation()

        self.assertEqual(0, rotation.xTheta)
        self.assertEqual(0, rotation.yTheta)
        self.assertEqual(0, rotation.zTheta)

    def testWhenAddOtherRotation_shouldAddItToCurrentRotation(self):
        rotation = Rotation(10, 30, 0)
        otherRotation = Rotation(90, 0, 90)

        rotation.add(otherRotation)

        self.assertEqual(10 + 90, rotation.xTheta)
        self.assertEqual(30 + 0, rotation.yTheta)
        self.assertEqual(0 + 90, rotation.zTheta)

    def testGivenNoRotation_whenAskedBoolean_shouldReturnFalse(self):
        noRotation = Rotation()
        self.assertFalse(noRotation)

    def testGivenRotation_whenAskedBoolean_shouldReturnTrue(self):
        noRotation = Rotation(10, 30, 0)
        self.assertTrue(noRotation)
    
    def testBetweenTwoIdenticalOrientations_shouldReturnNoRotation(self):
        orientation = Vector(0, 1, 1)
        rotation = Rotation.between(orientation, orientation)

        self.assertEqual(0, rotation.xTheta)
        self.assertEqual(0, rotation.yTheta)
        self.assertEqual(0, rotation.zTheta)
    
    def testBetweenOppositeSingleAxisOrientations_shouldReturnA180Rotation(self):
        fromOrientation = Vector(0, 0, 1)
        toOrientation = Vector(0, 0, -1)
        rotation = Rotation.between(fromOrientation, toOrientation)

        self.assertTrue((rotation.xTheta == 180) ^ (rotation.yTheta == 180))
        self.assertEqual(0, rotation.zTheta)

    def testBetweenTwoDifferentOrientations_shouldReturnExpectedRotation(self):
        fromOrientation = Vector(0, 1, 1)
        toOrientation = Vector(0, 1, -1)
        expectedRotation = Rotation(-90, 0, 0)

        rotation = Rotation.between(fromOrientation, toOrientation)

        self.assertEqual(expectedRotation.xTheta, rotation.xTheta)
        self.assertEqual(expectedRotation.yTheta, rotation.yTheta)
        self.assertEqual(expectedRotation.zTheta, rotation.zTheta)

    def testBetweenOppositePlanesOrientations_shouldReturnExpectedRotation(self):
        fromOrientation = Vector(1, 1, 1)
        toOrientation = Vector(-1, -1, -1)
        expectedRotation = Rotation(180, 0, -90)

        rotation = Rotation.between(fromOrientation, toOrientation)

        self.assertEqual(expectedRotation.xTheta, rotation.xTheta)
        self.assertEqual(expectedRotation.yTheta, rotation.yTheta)
        self.assertEqual(expectedRotation.zTheta, rotation.zTheta)
