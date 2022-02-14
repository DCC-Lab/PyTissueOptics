import unittest

from pytissueoptics.scene.geometry import BoundingBox


class TestBoundingBox(unittest.TestCase):
    def testWhenCreatedEmpty_shouldRaiseException(self):
        with self.assertRaises(Exception):
            bbox = BoundingBox()

    def testGivenLimits_whenAskingLimits_shouldReturnCorrectLimits(self):
        xLim = [0, 1]
        yLim = [-1, 0]
        zLim = [0.5, -0.5]
        bbox = BoundingBox(xLim, yLim, zLim)
        self.assertEqual(bbox.xLim, xLim)
        self.assertEqual(bbox.yLim, yLim)
        self.assertEqual(bbox.zLim, zLim)
