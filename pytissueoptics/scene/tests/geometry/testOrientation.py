import unittest

from pytissueoptics.scene.geometry import Orientation


class TestOrientation(unittest.TestCase):
    def testGivenNewOrientation_shouldBeAlignedWithAxes(self):
        orientation = Orientation()
        self.assertEqual(0, orientation.xTheta)
        self.assertEqual(0, orientation.yTheta)
        self.assertEqual(0, orientation.zTheta)

    def testWhenAddOtherOrientation_shouldAddItToCurrentOrientation(self):
        orientation = Orientation(10, 30, 0)
        otherOrientation = Orientation(90, 0, 90)

        orientation.add(otherOrientation)

        self.assertEqual(10+90, orientation.xTheta)
        self.assertEqual(30+0, orientation.yTheta)
        self.assertEqual(0+90, orientation.zTheta)
