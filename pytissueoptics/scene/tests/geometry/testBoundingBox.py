import unittest

from pytissueoptics.scene.geometry import BoundingBox, Vector


class TestBoundingBox(unittest.TestCase):
    
    def setUp(self):
        self.xLim = [0, 1]
        self.yLim = [-1, 0]
        self.zLim = [-0.5, 0.5]
    
    def testWhenCreatedEmpty_shouldRaiseException(self):
        with self.assertRaises(Exception):
            _ = BoundingBox()

    def testGivenLimits_whenAskingLimits_shouldReturnCorrectLimits(self):
        bbox = BoundingBox(self.xLim, self.yLim, self.zLim)
        self.assertEqual(bbox.xLim, self.xLim)
        self.assertEqual(bbox.yLim, self.yLim)
        self.assertEqual(bbox.zLim, self.zLim)

    def testGivenLimits_whenAskingMinMax_shouldReturnMinMax(self):
        bbox = BoundingBox(self.xLim, self.yLim, self.zLim)
        self.assertEqual(bbox.xMin, self.xLim[0])
        self.assertEqual(bbox.xMax, self.xLim[1])
        self.assertEqual(bbox.yMin, self.yLim[0])
        self.assertEqual(bbox.yMax, self.yLim[1])
        self.assertEqual(bbox.zMin, self.zLim[0])
        self.assertEqual(bbox.zMax, self.zLim[1])

    def testGivenVertices_whenAskingLimits_shouldReturnCorrectLimits(self):
        v1 = Vector(0, 1, 0)
        v2 = Vector(-1, 1, 2)
        v3 = Vector(-0.1, -1, 3.001)
        bbox = BoundingBox.fromVertices([v1, v2, v3])
        self.assertEqual(bbox.xLim, [-1, 0])
        self.assertEqual(bbox.yLim, [-1, 1])
        self.assertEqual(bbox.zLim, [0, 3.001])

    def testGivenWrongLimits_whenInit_shouldReturnMinMax(self):
        self.zLim = [0.5, -0.5]
        with self.assertRaises(ValueError):
            _ = BoundingBox(self.xLim, self.yLim, self.zLim)

    def testGivenLimits_whenChangeLimits_shouldReturnNewLimits(self):
        bbox = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox.change("x", "max", 3)
        self.assertEqual(bbox.xMax, 3)

    def testGivenLimits_whenChangeToWrongLimits_shouldRaiseValueError(self):
        bbox = BoundingBox(self.xLim, self.yLim, self.zLim)
        with self.assertRaises(ValueError):
            bbox.change("x", "max", -1)

    def testGiven2SimilarBBox_whenEquals_returnsTrue(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox2 = BoundingBox(self.xLim, self.yLim, self.zLim)
        self.assertTrue(bbox1 == bbox2)

    def testGiven2DifferentBBox_whenEquals_returnsFalse(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox2 = BoundingBox([0, 1.001], self.yLim, self.zLim)
        self.assertTrue(bbox1 != bbox2)