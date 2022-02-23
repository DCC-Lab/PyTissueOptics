import unittest
from mockito import mock, when

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.binary import MeanCentroidNodeSplitter, MiddlePolygonSpanNodeSplitter, \
    HardSAHNodeSplitter, BBoxPolyCounter
from pytissueoptics.scene.tree.treeConstructor import PolyCounter, SplitNodeResult
from pytissueoptics.scene.geometry import Vector


class TestBinaryMeanCentroidNodeSplitter(unittest.TestCase):
    def setUp(self):
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]

        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 3], zLim=[-1, 5])
        self.polyCounter = mock(PolyCounter)
        when(self.polyCounter).run(...).thenReturn([self.polygons])
        self.splitter = MeanCentroidNodeSplitter(self.polyCounter)

    def testOnXAXis_shouldReturnCorrectSplitNodeResult(self):
        splitNodeResult = self.splitter.run("x", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(False, "x", 7 / 9, [self.nodeBbox.changeToNew("x", "max", 7 / 9),
                                                               self.nodeBbox.changeToNew("x", "min", 7 / 9)],
                                           [self.polygons])
        self.assertEqual(validationResult, splitNodeResult)

    def testOnYAxis_shouldReturnCorrectSplitNodeResult(self):
        splitNodeResult = self.splitter.run("y", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(False, "y", 1, [self.nodeBbox.changeToNew("y", "max", 1),
                                                           self.nodeBbox.changeToNew("y", "min", 1)],
                                           [self.polygons])
        self.assertEqual(validationResult, splitNodeResult)


class TestBinaryMiddlePolygonSpanNodeSplitter(unittest.TestCase):
    def setUp(self):
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]

        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 3], zLim=[-1, 5])
        self.polyCounter = mock(PolyCounter)
        when(self.polyCounter).run(...).thenReturn([self.polygons])
        self.splitter = MiddlePolygonSpanNodeSplitter(self.polyCounter)

    def testOnXAxis_shouldReturnCorrectSplitNodeResult(self):
        splitNodeResult = self.splitter.run("x", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(False, "x", 1, [self.nodeBbox.changeToNew("x", "max", 1),
                                                           self.nodeBbox.changeToNew("x", "min", 1)],
                                           [self.polygons])
        self.assertEqual(validationResult, splitNodeResult)

    def testOnYAxis_shouldReturnCorrectSplitNodeResult(self):
        splitNodeResult = self.splitter.run("y", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(False, "y", 1 / 2, [self.nodeBbox.changeToNew("y", "max", 1 / 2),
                                                               self.nodeBbox.changeToNew("y", "min", 1 / 2)],
                                           [self.polygons])
        self.assertEqual(validationResult, splitNodeResult)


class TestBinaryHardSAHNodeSplitter(unittest.TestCase):
    def setUp(self):
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]

        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 3], zLim=[-1, 5])
        self.polyCounter = BBoxPolyCounter()
        self.splitter = HardSAHNodeSplitter(self.polyCounter, nbOfSplitPlanes=3, splitCostPercentage=0.2)

    def testOnXAxis_givenSplitCostOf20Percent_shouldReturnCorrectSplitNodeResult(self):
        splitNodeResult = self.splitter.run("x", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(False, "x", 1.5, [self.nodeBbox.changeToNew("x", "max", 1.5),
                                                             self.nodeBbox.changeToNew("x", "min", 1.5)],
                                           [[self.polygons[0], self.polygons[1]], [self.polygons[2]]])
        self.assertEqual(validationResult, splitNodeResult)

    def testOnXAxis_givenSplitCostOf40Percent_shouldReturnStopCondition(self):
        self.splitter = HardSAHNodeSplitter(self.polyCounter, nbOfSplitPlanes=3, splitCostPercentage=0.4)
        splitNodeResult = self.splitter.run("x", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(True, "x", 1.5, [self.nodeBbox.changeToNew("x", "max", 1.5),
                                                            self.nodeBbox.changeToNew("x", "min", 1.5)],
                                           [[self.polygons[0], self.polygons[1]], [self.polygons[2]]])
        self.assertEqual(validationResult, splitNodeResult)

    def testOnYAxis_givenSplitCostOf20Percent_shouldReturnStopCondition(self):
        splitNodeResult = self.splitter.run("y", self.nodeBbox, self.polygons)
        validationResult = SplitNodeResult(True, "y", 2.0, [self.nodeBbox.changeToNew("y", "max", 2),
                                                            self.nodeBbox.changeToNew("y", "min", 2)],
                                           [[self.polygons[0], self.polygons[1], self.polygons[2]], [self.polygons[2]]])
        self.assertEqual(validationResult, splitNodeResult)
