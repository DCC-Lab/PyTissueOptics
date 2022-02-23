import unittest

from pytissueoptics.scene.geometry import Polygon, Vector, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.binary import RotatingAxis, LargestSpanAxis, LargestPolygonSpanAxis


class TestBinaryRotateAxisSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]

        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 3], zLim=[-1, 5])

    def testRotateAxis_givenDifferentDepth_shouldReturnCorrectAxis(self):
        axisSelector = RotatingAxis()
        axis = axisSelector.select(0, self.nodeBbox, self.polygons)
        self.assertEqual("x", axis)
        axis = axisSelector.select(1, self.nodeBbox, self.polygons)
        self.assertEqual("y", axis)
        axis = axisSelector.select(2, self.nodeBbox, self.polygons)
        self.assertEqual("z", axis)
        axis = axisSelector.select(3, self.nodeBbox, self.polygons)
        self.assertEqual("x", axis)


class TestBinaryLargestPolygonSpanSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]

        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 3], zLim=[-1, 5])

    def testLargestPolygonSpanAxis_givenDifferentPolygons_shouldReturnCorrectAxis(self):
        axisSelector = LargestPolygonSpanAxis()
        axis = axisSelector.select(0, self.nodeBbox, self.polygons)
        self.assertEqual("y", axis)

    def testLargestPolygonSpanAxis_givenSamePolygonsSpan_shouldReturnFirstAxis(self):
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -1, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]
        axisSelector = LargestPolygonSpanAxis()
        axis = axisSelector.select(0, self.nodeBbox, self.polygons)
        self.assertEqual("x", axis)


class TestBinaryLargestSpanAxiSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]

        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 3], zLim=[-1, 5])

    def testLargestSpanAxis_givenDifferentPolygons_shouldReturnCorrectAxis(self):
        axisSelector = LargestSpanAxis()
        axis = axisSelector.select(0, self.nodeBbox, self.polygons)
        self.assertEqual("z", axis)

    def testLargestSpanAxis_givenSamePolygonsSpan_shouldReturnFirstAxis(self):
        self.nodeBbox = BoundingBox(xLim=[-1, 4], yLim=[-1, 5], zLim=[-1, 5])
        axisSelector = LargestSpanAxis()
        axis = axisSelector.select(0, self.nodeBbox, self.polygons)
        self.assertEqual("y", axis)
