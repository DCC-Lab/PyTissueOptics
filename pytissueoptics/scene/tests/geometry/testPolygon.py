import unittest

from pytissueoptics.scene.geometry import Polygon, Vector, Vertex


class TestPolygon(unittest.TestCase):
    def testGivenANewPolygon_shouldDefineItsNormal(self):
        polygon = Polygon(vertices=[Vertex(0, 0, 0), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0)])
        self.assertEqual(Vertex(0, 0, 1), polygon.normal)

    def testGivenANewPolygonWithNormal_shouldUseProvidedNormalWithoutNormalizing(self):
        forcedNormal = Vector(7, 5, 3)
        polygon = Polygon(vertices=[Vertex(0, 0, 0), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0)],
                          normal=forcedNormal)
        self.assertEqual(forcedNormal, polygon.normal)

    def testGivenANewPolygon_shouldDefineItsCentroid(self):
        polygon = Polygon(vertices=[Vertex(0, 0, 1), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0)])
        self.assertEqual(Vector(5/4, 3/4, 1/4), polygon.centroid)

    def testGivenANewPolygon_whenModifyingVertexAndResetBoundingBox_shouldChangeBbox(self):
        triangle = Polygon(vertices=[Vertex(0, 0, 0), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 4, 2)])
        oldBbox = triangle.bbox
        triangle.vertices[0].update(5, 1, 1)
        triangle.resetBoundingBox()
        newBbox = triangle.bbox
        self.assertNotEqual(oldBbox, newBbox)

    def testGiven2EqualPolygons_whenEquals_shouldReturnTrue(self):
        polygon1 = Polygon(vertices=[Vertex(0, 0, 0), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0)])
        polygon2 = Polygon(vertices=[Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0), Vertex(0, 0, 0)])
        self.assertEqual(polygon1, polygon2)

    def testGiven2DifferentPolygons_whenEquals_shouldReturnFalse(self):
        polygon1 = Polygon(vertices=[Vertex(0, 0, 0), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0)])
        polygon2 = Polygon(vertices=[Vertex(1, 0, 0), Vertex(2, 2, 0), Vertex(1, 1, 0), Vertex(0, 0, 0)])
        self.assertNotEqual(polygon1, polygon2)

    def testWhenGetCentroid_shouldReturnAverageVertex(self):
        polygon = Polygon(vertices=[Vertex(0, 0, 0), Vertex(2, 0, 0), Vertex(2, 2, 0), Vertex(0, 2, 0)])
        self.assertEqual(Vector(1, 1, 0), polygon.getCentroid())
