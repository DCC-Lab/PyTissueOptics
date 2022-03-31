import unittest

from pytissueoptics.scene.geometry import Polygon, Vector


class TestPolygon(unittest.TestCase):
    def testGivenANewPolygon_shouldDefineItsNormal(self):
        polygon = Polygon(vertices=[Vector(0, 0, 0), Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0)])
        self.assertEqual(Vector(0, 0, 1), polygon.normal)

    def testGivenANewPolygonWithNormal_shouldUseProvidedNormalWithoutNormalizing(self):
        forcedNormal = Vector(7, 5, 3)
        polygon = Polygon(vertices=[Vector(0, 0, 0), Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0)],
                          normal=forcedNormal)
        self.assertEqual(forcedNormal, polygon.normal)

    def testGivenANewPolygon_shouldDefineItsCentroid(self):
        polygon = Polygon(vertices=[Vector(0, 0, 1), Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0)])
        self.assertEqual(Vector(5/4, 3/4, 1/4), polygon.centroid)

    def testGivenANewPolygon_whenModifyingVertexAndResetBoundingBox_shouldChangeBbox(self):
        triangle = Polygon(vertices=[Vector(0, 0, 0), Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 4, 2)])
        oldBbox = triangle.bbox
        triangle.vertices[0].update(5, 1, 1)
        triangle.resetBoundingBox()
        newBbox = triangle.bbox
        self.assertNotEqual(oldBbox, newBbox)

    def testGiven2EqualPolygons_whenEquals_shouldReturnTrue(self):
        polygon1 = Polygon(vertices=[Vector(0, 0, 0), Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0)])
        polygon2 = Polygon(vertices=[Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0), Vector(0, 0, 0)])
        self.assertEqual(polygon1, polygon2)

    def testGiven2DifferentPolygons_whenEquals_shouldReturnFalse(self):
        polygon1 = Polygon(vertices=[Vector(0, 0, 0), Vector(2, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0)])
        polygon2 = Polygon(vertices=[Vector(1, 0, 0), Vector(2, 2, 0), Vector(1, 1, 0), Vector(0, 0, 0)])
        self.assertNotEqual(polygon1, polygon2)

    def testWhenGetCentroid_shouldReturnAverageVertex(self):
        polygon = Polygon(vertices=[Vector(0, 0, 0), Vector(2, 0, 0), Vector(2, 2, 0), Vector(0, 2, 0)])
        self.assertEqual(Vector(1, 1, 0), polygon.getCentroid())
