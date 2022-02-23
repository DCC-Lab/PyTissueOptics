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

    def testGiven2SimilarBBox_whenEquals_shouldReturnTrue(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox2 = BoundingBox(self.xLim, self.yLim, self.zLim)
        self.assertTrue(bbox1 == bbox2)

    def testGiven2DifferentBBox_whenEquals_shouldReturnFalse(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox2 = BoundingBox([0, 1.001], self.yLim, self.zLim)
        self.assertTrue(bbox1 != bbox2)

    def testGivenNewBBoxFromVertices_shouldDefineBoundingBoxAroundVertices(self):
        v1 = Vector(0, 1, 0)
        v2 = Vector(-1, 1, 2)
        v3 = Vector(-0.1, -1, 3.001)
        bbox = BoundingBox.fromVertices([v1, v2, v3])

        self.assertEqual(bbox.xLim, [-1, 0])
        self.assertEqual(bbox.yLim, [-1, 1])
        self.assertEqual(bbox.zLim, [0, 3.001])

    def testGivenNewBBox_shouldDefineWidths(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        self.assertEqual(1, bbox1.xWidth)
        self.assertEqual(1, bbox1.yWidth)
        self.assertEqual(2, bbox1.zWidth)

    def testGivenNewBBox_shouldDefineArea(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        expectedArea = (1 * 1) * 2 + (2 * 1) * 4
        self.assertEqual(expectedArea, bbox1.getArea())

    def testGivenAContainedPoint_whenContains_shouldReturnTrue(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        point1 = Vector(0.1, -0.1, -0.9)
        self.assertEqual(True, bbox1.contains(point1))

    def testGivenAnOutsidePoint_whenContains_shouldReturnFalse(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        point1 = Vector(0.1, -0.1, -1.1)
        self.assertEqual(False, bbox1.contains(point1))

    def testGivenATouchingPoint_whenContains_shouldReturnFalse(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        point1 = Vector(0, 0, -1)
        self.assertEqual(False, bbox1.contains(point1))

    def testGivenABBox_whenExtendTo_shouldIncreaseTheBboxLimits(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        bbox2 = BoundingBox([0, 0.1], [-2, 2], [-2, 0.9])
        bbox1.extendTo(bbox2)
        expectedBbox = BoundingBox([0, 1], [-2, 2], [-2, 1])
        self.assertEqual(expectedBbox, bbox1)
