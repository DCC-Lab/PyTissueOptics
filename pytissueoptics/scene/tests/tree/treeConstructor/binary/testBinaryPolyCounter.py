import unittest

from pytissueoptics.scene.geometry import Polygon, Vector, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.binary import BBoxPolyCounter, CentroidPolyCounter


class TestBinaryPolyCounter(unittest.TestCase):
    def setUp(self) -> None:
        self.polygons = [Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0)]),
                         Polygon(vertices=[Vector(0, 0, 0), Vector(0, 1, 0), Vector(-1, -2, 0)]),
                         Polygon(vertices=[Vector(2, 2, 2), Vector(3, 3, 3), Vector(2, 3, 2)])]


    def testBboxPolyCounter_givenLineAndAxis_shouldReturnSeparatedPolygons(self):
        polygonCounter = BBoxPolyCounter()
        polygonGroups = polygonCounter.run(1.5, "x", self.polygons)
        verificationGroups = [[self.polygons[0], self.polygons[1]], [self.polygons[2]]]
        self.assertEqual(verificationGroups, polygonGroups)

    def testBboxPolyCounter_givenLineAxisWithSharedPolygons_shouldReturnSharedPolygons(self):
        polygonCounter = BBoxPolyCounter()
        polygonGroups = polygonCounter.run(2, "x", self.polygons)
        verificationGroups = [self.polygons, [self.polygons[2]]]
        self.assertEqual(verificationGroups, polygonGroups)

    def testBboxPolyCounter_givenLineAxisWithAllPolygons_shouldReturnAnEmptyGroup(self):
        polygonCounter = BBoxPolyCounter()
        polygonGroups = polygonCounter.run(3.1, "y", self.polygons)
        verificationGroups = [self.polygons, []]
        self.assertEqual(verificationGroups, polygonGroups)

    def testCentroidPolyCounter_givenLineAndAxis_shouldReturnSeparatedPolygons(self):
        polygonCounter = CentroidPolyCounter()
        polygonGroups = polygonCounter.run(1.5, "x", self.polygons)
        verificationGroups = [[self.polygons[0], self.polygons[1]], [self.polygons[2]]]
        self.assertEqual(verificationGroups, polygonGroups)

    def testCentroidPolyCounter_givenLineCloseToCentroid_shouldReturnSeparatedPolygons(self):
        polygonCounter = CentroidPolyCounter()
        polygonGroups = polygonCounter.run(2.3, "x", self.polygons)
        verificationGroups = [[self.polygons[0], self.polygons[1]], [self.polygons[2]]]
        self.assertEqual(verificationGroups, polygonGroups)

    def testCentroidPolyCounter_givenLineOnCentroid_shouldReturnSharedPolygons(self):
        polygonCounter = CentroidPolyCounter()
        polygonGroups = polygonCounter.run(7/3, "x", self.polygons)
        verificationGroups = [self.polygons, [self.polygons[2]]]
        self.assertEqual(verificationGroups, polygonGroups)
