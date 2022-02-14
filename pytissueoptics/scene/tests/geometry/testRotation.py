import unittest

from pytissueoptics.scene.geometry import Rotation


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

        self.assertEqual(10+90, rotation.xTheta)
        self.assertEqual(30+0, rotation.yTheta)
        self.assertEqual(0+90, rotation.zTheta)

    def testGivenNoRotation_whenAskedBoolean_shouldReturnFalse(self):
        noRotation = Rotation()
        self.assertFalse(noRotation)

    def testGivenRotation_whenAskedBoolean_shouldReturnTrue(self):
        noRotation = Rotation(10, 30, 0)
        self.assertTrue(noRotation)

    def testWhenGetInverse_shouldReturnNewNegativeRotation(self):
        # fixme: negative rotation is the inverse ! (reverse order ! => ask utils)
        rotation = Rotation(10, 30, 0)
        inverseRotation = rotation.getInverse()

        self.assertEqual(-10, inverseRotation.xTheta)
        self.assertEqual(-30, inverseRotation.yTheta)
        self.assertEqual(0, inverseRotation.zTheta)
