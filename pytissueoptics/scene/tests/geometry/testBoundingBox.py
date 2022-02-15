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
