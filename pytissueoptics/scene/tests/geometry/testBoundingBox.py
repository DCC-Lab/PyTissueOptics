import unittest

from pytissueoptics.scene.geometry import BoundingBox, Vector


class TestBoundingBox(unittest.TestCase):
    def testWhenCreatedEmpty_shouldRaiseException(self):
        with self.assertRaises(Exception):
            _ = BoundingBox()

    def testGivenLimits_whenAskingLimits_shouldReturnCorrectLimits(self):
        xLim = [0, 1]
        yLim = [-1, 0]
        zLim = [0.5, -0.5]
        bbox = BoundingBox(xLim, yLim, zLim)
        self.assertEqual(bbox.xLim, xLim)
        self.assertEqual(bbox.yLim, yLim)
        self.assertEqual(bbox.zLim, zLim)

    def testGivenLimits_whenAskingMinMax_shouldReturnMinMax(self):
        xLim = [0, 1]
        yLim = [-1, 0]
        zLim = [-0.5, 0.5]
        bbox = BoundingBox(xLim, yLim, zLim)
        self.assertEqual(bbox.xMin, xLim[0])
        self.assertEqual(bbox.xMax, xLim[1])
        self.assertEqual(bbox.yMin, yLim[0])
        self.assertEqual(bbox.yMax, yLim[1])
        self.assertEqual(bbox.zMin, zLim[0])
        self.assertEqual(bbox.zMax, zLim[1])

    def testGivenVertices_whenAskingLimits_shouldReturnCorrectLimits(self):
        v1 = Vector(0, 1, 0)
        v2 = Vector(-1, 1, 2)
        v3 = Vector(-0.1, -1, 3.001)
        bbox = BoundingBox.fromVertices([v1, v2, v3])
        self.assertEqual(bbox.xLim, [-1, 0])
        self.assertEqual(bbox.yLim, [-1, 1])
        self.assertEqual(bbox.zLim, [0, 3.001])

    def testGivenWrongLimits_whenInit_shouldReturnMinMax(self):
        xLim = [0, 1]
        yLim = [-1, 0]
        zLim = [0.5, -0.5]
        with self.assertRaises(ValueError):
            _ = BoundingBox(xLim, yLim, zLim)

    