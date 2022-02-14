import unittest

from pytissueoptics.scene.geometry import BoundingBox, Vector


class TestBoundingBox(unittest.TestCase):
    
    def setUp(self):
        self.xLim = [0, 1]
        self.yLim = [-1, 0]
        self.zLim = [-0.5, 0.5]
    
    def testGivenNoLimits_shouldRaiseException(self):
        with self.assertRaises(Exception):
            _ = BoundingBox()

    def testGivenWrongLimits_shouldRaiseValueError(self):
        self.zLim = [0.5, -0.5]
        with self.assertRaises(ValueError):
            _ = BoundingBox(self.xLim, self.yLim, self.zLim)

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

    def testGivenNewBBoxFromVertices_whenAskingLimits_shouldReturnCorrectLimits(self):
        v1 = Vector(0, 1, 0)
        v2 = Vector(-1, 1, 2)
        v3 = Vector(-0.1, -1, 3.001)
        bbox = BoundingBox.fromVertices([v1, v2, v3])

        self.assertEqual(bbox.xLim, [-1, 0])
        self.assertEqual(bbox.yLim, [-1, 1])
        self.assertEqual(bbox.zLim, [0, 3.001])

    def testGiven2SimilarBBox_whenEquals_shouldReturnTrue(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox2 = BoundingBox(self.xLim, self.yLim, self.zLim)
        self.assertTrue(bbox1 == bbox2)

    def testGiven2DifferentBBox_whenEquals_shouldReturnFalse(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox2 = BoundingBox([0, 1.001], self.yLim, self.zLim)
        self.assertTrue(bbox1 != bbox2)
