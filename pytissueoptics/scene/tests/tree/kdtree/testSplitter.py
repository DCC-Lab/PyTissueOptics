import unittest
from pytissueoptics.scene.geometry.tree.treeStrategy import MeanCentroidSplitter, BinarySplitter
from pytissueoptics.scene.geometry import Vector, Triangle


class TestMeanCentroidSplitter(unittest.TestCase):
    def setUp(self):
        self._splitter = MeanCentroidSplitter()

    def testGivenAxisAndPolygons_shouldSplitAtMeanCentroidOfThatAxis(self):
        polygon1 = Triangle(Vector(1, 1, 2), Vector(0, 0, 1), Vector(-1, -2, 1))
        polygon2 = Triangle(Vector(-4, -6, -6), Vector(-4, -3, -3), Vector(-4, -5, -4))
        xSplitValue = self._splitter.calculateSplitLine([polygon1, polygon2], "x")
        ySplitValue = self._splitter.calculateSplitLine([polygon1, polygon2], "y")
        zSplitValue = self._splitter.calculateSplitLine([polygon1, polygon2], "z")
        self.assertEqual(xSplitValue, -2)
        self.assertEqual(ySplitValue, -2.5)
        self.assertEqual(zSplitValue, -1.5)


class TestBinarySplitter(unittest.TestCase):

    def setUp(self):
        self._splitter = BinarySplitter()

    def testGivenAxisAndPolygons_shouldSplitInTheMiddle(self):
        polygon1 = Triangle(Vector(1, 1, 2), Vector(0, 0, 1), Vector(-1, -2, 1))
        polygon2 = Triangle(Vector(-4, -6, -6), Vector(-4, -3, -3), Vector(-4, -5, -4))
        xSplitValue = self._splitter.calculateSplitLine([polygon1, polygon2], "x")
        ySplitValue = self._splitter.calculateSplitLine([polygon1, polygon2], "y")
        zSplitValue = self._splitter.calculateSplitLine([polygon1, polygon2], "z")
        self.assertEqual(xSplitValue, -1.5)
        self.assertEqual(ySplitValue, -2.5)
        self.assertEqual(zSplitValue, -2)
