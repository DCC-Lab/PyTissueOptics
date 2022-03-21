import unittest
from typing import Tuple
from math import sqrt

from pytissueoptics.scene.geometry import Triangle, Quad, Polygon, Vector
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree.treeConstructor.binary import FastBinaryTreeConstructor



class TestFastBinaryTreeConstructor(unittest.TestCase):
    def setUp(self) -> None:
        self._fbtc = FastBinaryTreeConstructor()

    @staticmethod
    def _makeSplitPlane(splitAxis: str, splitValue: float) -> Tuple[Vector, Vector]:
        if splitAxis == "x":
            normal = Vector(1, 0, 0)
            planePoint = Vector(splitValue, 0, 0)
            return normal, planePoint
        elif splitAxis == "y":
            normal = Vector(0, 1, 0)
            planePoint = Vector(0, splitValue, 0)
            return normal, planePoint
        elif splitAxis == "z":
            normal = Vector(0, 0, 1)
            planePoint = Vector(0, 0, splitValue)
            return normal, planePoint
        
    def test_givenPolygons_whenClassifying_shouldReturnCorrect3Groups(self):
        rightTriangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0))
        quad = Quad(Vector(-5, -5, 0), Vector(-5, 5, 0), Vector(5, 5, 0), Vector(5, -5, 0))
        polygon = Polygon(vertices=[Vector(3, 3, 2), Vector(1, 1, 1), Vector(1, -1, 1)])
        leftPolygon = Polygon(vertices=[Vector(3, -3, 2), Vector(1, -1, -0.5), Vector(1, -1, -1)])

        toClassify = [rightTriangle, quad, polygon, leftPolygon]
        left, right, both = self._fbtc._classifyPolygons(-0.5, "y", toClassify)
        self.assertListEqual(left, [leftPolygon])
        self.assertListEqual(right, [rightTriangle])
        self.assertListEqual(both, [quad, polygon])

    def test_givenATriangleAndAPlane_whenPolygonAsRays_shouldReturnCorrectRays(self):
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

    def test_givenAPolygonAndAPlane_whenSplittingPolygon_shouldReturn2Polygons(self):
        toBeSplitted = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)])]
        splitValue = 0.5
        splitAxis = "x"
        normal, dot = self._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitPolygons(toBeSplitted, normal, dot, splitAxis, splitValue)
        expectedLeft = [Polygon(vertices=[Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), Vector(0.5, -0.5, 0.5)])]
        expectedRight = [
            Polygon(vertices=[Vector(1, 1, 1), Vector(0.5, 0.5, 0.5), Vector(0.5, -0.5, 0.5), Vector(1, -1, 1)])]

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right[0], expectedRight[0])

    def test_givenAPolygonAndAPlane_whenSplittingOnAVertexInside_shouldReturn2Polygons(self):
        toBeSplitted = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)])]
        splitValue = 0
        splitAxis = "y"
        normal, dot = self._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitPolygons(toBeSplitted, normal, dot, splitAxis, splitValue)
        expectedLeft = [Polygon(vertices=[Vector(1, -1, 1), Vector(1, 0, 1), Vector(0, 0, 0)])]
        expectedRight = [Polygon(vertices=[Vector(1, 1, 1), Vector(1, 0, 1), Vector(0, 0, 0)])]
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right[0], expectedRight[0])

    def test_givenAPolygonAndAPlane_whenSplittingOnAVertexOutside_shouldReturn1Polygons(self):
        toBeSplitted = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)])]
        splitValue = 1
        splitAxis = "y"
        normal, dot = self._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitPolygons(toBeSplitted, normal, dot, splitAxis, splitValue)
        expectedLeft = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)])]
        expectedRight = []
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right, expectedRight)

    def test_givenAPolygonAndAPlane_whenSplittingOn2VerticesOutside_shouldReturn1Polygons(self):
        toBeSplitted = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)])]
        splitValue = 1
        splitAxis = "x"
        normal, dot = self._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitPolygons(toBeSplitted, normal, dot, splitAxis, splitValue)
        expectedLeft = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)])]
        expectedRight = []
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right, expectedRight)

    def test_givenAPolygonAndAPlane_whenSplittingOn2VerticesInside_shouldReturn2Polygons(self):
        toBeSplitted = [Polygon(vertices=[Vector(0, 0, 0), Vector(1, 1, 1), Vector(2, 0, 1), Vector(1, -1, 1)])]
        splitValue = 1
        splitAxis = "x"
        normal, dot = self._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitPolygons(toBeSplitted, normal, dot, splitAxis, splitValue)
        expectedLeft = [Polygon(vertices=[Vector(1, -1, 1), Vector(1, 1, 1), Vector(0, 0, 0)])]
        expectedRight = [Polygon(vertices=[Vector(1, -1, 1), Vector(1, 1, 1), Vector(2, 0, 1)])]
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right[0], expectedRight[0])

    def test_givenUltraThinPolygon_whenSplitting_shouldStillReturn2Polygons(self):
        vertices = [Vector(8.860660171779822, 5.000000000000001, -4.9455),
                    Vector(8.856599089933916, 4.995938918154095, -4.9455),
                    Vector(8.856599089933916, 4.99986899735981, -4.9455)]
        toBeSplit = [Polygon(vertices=vertices)]
        splitAxis = "y"
        splitValue = 4.999868997359811
        normal, dot = self._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitPolygons(toBeSplit, normal, dot, splitAxis, splitValue)
        self.assertEqual(1, len(left))
        self.assertEqual(1, len(right))


