import unittest

from pytissueoptics.scene.geometry import BoundingBox, Vector, Polygon


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

    def testGivenNewBBoxFromPolygons_shouldDefineBoundingBoxAroundPolygons(self):
        polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 2, 1), Vector(1, -1, 1)]),
                    Polygon(vertices=[Vector(0, 0, 0), Vector(-1, -1, -1), Vector(-2, -2, -3)])]
        bbox = BoundingBox.fromPolygons(polygons)

        self.assertEqual(bbox.xLim, [-2, 1])
        self.assertEqual(bbox.yLim, [-2, 2])
        self.assertEqual(bbox.zLim, [-3, 1])

    def testGivenNewBBox_shouldDefineWidths(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        self.assertEqual(1, bbox1.xWidth)
        self.assertEqual(1, bbox1.yWidth)
        self.assertEqual(2, bbox1.zWidth)

    def testGivenNewBBox_shouldDefineArea(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        expectedArea = (1 * 1) * 2 + (2 * 1) * 4
        self.assertEqual(expectedArea, bbox1.getArea())

    def testGivenNewBBox_shouldDefineCenter(self):
        bbox = BoundingBox(self.xLim, self.yLim, self.zLim)
        expectedCenter = Vector(0.5, -0.5, 0)
        self.assertEqual(expectedCenter, bbox.center)

    def testGivenNewBBox_whenUpdate_shouldChangeTheDesiredLimitValue(self):
        bbox = BoundingBox(self.xLim, self.yLim, self.zLim)
        bbox.update("x", "min", -10)
        self.assertEqual([-10, 1], bbox.xLim)

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

    def testGivenABBox_whenShrinkTo_shouldDecreaseTheBboxLimits(self):
        bbox1 = BoundingBox(self.xLim, self.yLim, [-1, 1])
        bbox2 = BoundingBox([0, 0.1], [-2, 2], [-2, 0.9])
        bbox1.shrinkTo(bbox2)
        expectedBbox = BoundingBox([0, 0.1], [-1, 0], [-1, 0.9])
        self.assertEqual(expectedBbox, bbox1)

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
