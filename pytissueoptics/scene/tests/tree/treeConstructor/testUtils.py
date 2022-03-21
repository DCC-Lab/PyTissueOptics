import unittest
from pytissueoptics.scene.geometry import Polygon, Vector
from pytissueoptics.scene.tree.treeConstructor.utils import meanCentroid


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
