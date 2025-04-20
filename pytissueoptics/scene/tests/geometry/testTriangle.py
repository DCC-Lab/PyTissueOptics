import unittest

from pytissueoptics.scene.geometry import Triangle, Vector, Vertex


class TestTriangle(unittest.TestCase):
    def testGivenANewTriangle_shouldDefineItsNormal(self):
        triangle = Triangle(v1=Vertex(0, 0, 0), v2=Vertex(2, 0, 0), v3=Vertex(2, 2, 0))
        self.assertEqual(Vector(0, 0, 1), triangle.normal)

    def testGivenANewTriangle_shouldDefineItsCentroid(self):
        triangle = Triangle(v1=Vertex(0, 0, 1), v2=Vertex(2, 0, 0), v3=Vertex(2, 2, 0))
        self.assertEqual(Vector(4 / 3, 2 / 3, 1 / 3), triangle.centroid)

    def testGivenANewTriangle_whenModifyingVertex_resetBoundingBoxShouldChangeBbox(self):
        triangle = Triangle(v1=Vertex(0, 0, 0), v2=Vertex(2, 0, 0), v3=Vertex(2, 2, 0))
        oldBbox = triangle.bbox

        triangle.vertices[0].update(1, 1, 1)
        triangle.resetBoundingBox()
        newBbox = triangle.bbox

        self.assertNotEqual(oldBbox, newBbox)
