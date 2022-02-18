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

    def testWhenIntersectsWithIntersectingBBox_shouldReturnTrue(self):
        bbox = BoundingBox([0, 5], [0, 5], [0, 5])
        partiallyIntersectingBox = BoundingBox([2, 6], [2, 6], [2, 6])
        insideBox = BoundingBox([2, 3], [2, 3], [2, 3])
        outsideBox = BoundingBox([-1, 6], [-1, 6], [-1, 6])
        sameBox = BoundingBox([0, 5], [0, 5], [0, 5])
        touchingBox = BoundingBox([5, 6], [5, 6], [5, 6])

        self.assertTrue(bbox.intersects(partiallyIntersectingBox))
        self.assertTrue(bbox.intersects(insideBox))
        self.assertTrue(bbox.intersects(outsideBox))
        self.assertTrue(bbox.intersects(sameBox))
        self.assertTrue(bbox.intersects(touchingBox))

    def testWhenIntersectsWithNonIntersectingBBox_shouldReturnFalse(self):
        bbox = BoundingBox([0, 5], [0, 5], [0, 5])
        nonIntersectingBox = BoundingBox([6, 7], [6, 7], [6, 7])

        self.assertFalse(bbox.intersects(nonIntersectingBox))
