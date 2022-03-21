import unittest
from pytissueoptics.scene.geometry import Polygon, Vector, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.utils import getPolygonsBbox, meanCentroid


class TestTreeConstructorUtils(unittest.TestCase):
    def testGivenListOfPolygonAndAxis_whenMeanCentroid_shouldReturnCentroidOnThisAxis(self):
        polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, 0, 0)]),
                    Polygon(vertices=[Vector(0, 0, 0), Vector(-1, -1, 1), Vector(-1, 0, 0)])]
        xMeanCentroid = meanCentroid("x", polygons)
        yMeanCentroid = meanCentroid("y", polygons)
        zMeanCentroid = meanCentroid("z", polygons)
        self.assertEqual(0, xMeanCentroid)
        self.assertEqual(0, yMeanCentroid)
        self.assertEqual(1/3, zMeanCentroid)

    def testGivenListOfPolygon_whenGetPolygonsBbox_shouldReturnBboxContainingAllPolygons(self):
        polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, 0, 0)]),
                    Polygon(vertices=[Vector(0, 0, 0), Vector(-1, -1, 1), Vector(-1, 0, 0)]),
                    Polygon(vertices=[Vector(5, 5, 0), Vector(-1, -1, 1), Vector(-1, 0, 0)])]
        bbox = getPolygonsBbox(polygons)
        expectedBbox = BoundingBox([-1, 5], [-1, 5], [0, 1])
        self.assertEqual(expectedBbox, bbox)
