import unittest
from math import sqrt
from pytissueoptics.scene.geometry import Triangle, Quad, Polygon, Vector
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree.treeConstructor.binary import FastBinaryTreeConstructor


class TestFastBinaryTreeConstructor(unittest.TestCase):
    def setUp(self) -> None:
        self._fbtc = FastBinaryTreeConstructor()

    def testGetPolygonsAsRays(self):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0))
        quad = Quad(Vector(-5, -5, 0), Vector(-5, 5, 0), Vector(5, 5, 0), Vector(5, -5, 0))
        triangleRays = self._fbtc._getPolygonAsRays(triangle)
        quadRays = self._fbtc._getPolygonAsRays(quad)
        expectedTriangleRays = [Ray(Vector(0, 0, 0), Vector(1, 0, 0), 1.0),
                                Ray(Vector(1, 0, 0), Vector(0, 1, 0), 1.0),
                                Ray(Vector(1, 1, 0), Vector(-sqrt(2)/2, -sqrt(2)/2, 0), sqrt(2))]
        expectedQuadRays = [Ray(Vector(-5, -5, 0), Vector(0, 1, 0), 10.0),
                            Ray(Vector(-5, 5, 0), Vector(1, 0, 0), 10.0),
                            Ray(Vector(5, 5, 0), Vector(0, -1, 0), 10.0),
                            Ray(Vector(5, -5, 0), Vector(-1, 0, 0), 10.0)]

        for i, (tri, quad) in enumerate(zip(triangleRays, quadRays)):
            self.assertEqual(tri.direction, expectedTriangleRays[i].direction, 2)
            self.assertEqual(tri.origin, expectedTriangleRays[i].origin, 2)
            self.assertEqual(tri.length, expectedTriangleRays[i].length, 2)
            self.assertEqual(quad.direction, expectedQuadRays[i].direction, 2)
            self.assertEqual(quad.origin, expectedQuadRays[i].origin, 2)
            self.assertEqual(quad.length, expectedQuadRays[i].length, 2)

    def testSplitPolygons(self):
        pass