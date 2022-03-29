import unittest
from math import sqrt

from pytissueoptics.scene import Material
from pytissueoptics.scene.geometry import Triangle, Vector, BoundingBox
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor.binary import SplitThreeAxesConstructor, SAHSearchResult


class TestSplitConstructor(unittest.TestCase):
    def setUp(self) -> None:
        self._fbtc = SplitThreeAxesConstructor()

    def test_givenATriangleAndAPlane_whenPolygonAsRays_shouldReturnCorrectRays(self):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0))
        triangleRays = self._fbtc._getPolygonAsRays(triangle)
        expectedTriangleRays = [Ray(Vector(0, 0, 0), Vector(1, 0, 0), 1.0),
                                Ray(Vector(1, 0, 0), Vector(0, 1, 0), 1.0),
                                Ray(Vector(1, 1, 0), Vector(-sqrt(2) / 2, -sqrt(2) / 2, 0), sqrt(2))]

        for i, tri in enumerate(triangleRays):
            self.assertEqual(tri.direction, expectedTriangleRays[i].direction, 2)
            self.assertEqual(tri.origin, expectedTriangleRays[i].origin, 2)
            self.assertEqual(tri.length, expectedTriangleRays[i].length, 2)

    def test_givenAPolygonAndAPlane_whenSplittingPolygon_shouldReturn3Polygons(self):
        toBeSplit = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1))]
        splitValue = 0.5
        splitAxis = "x"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitTriangles(normal, dot)
        expectedLeft = [Triangle(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), Vector(0.5, -0.5, 0.5))]

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(len(right), 2)

    def test_givenAPolygonAndAPlane_whenSplittingPolygon_splitPolygonShouldHaveGoodMaterialAndNormal(self):
        myMaterial = Material(1, 1, 0.5)
        toBeSplit = [Triangle(Vector(0, 0, 0), Vector(0, 1, 0), Vector(0, 1, 1), insideMaterial=myMaterial)]
        splitValue = 0.5
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitTriangles(normal, dot)
        expectedLeft = [Triangle(Vector(0, 0, 0), Vector(0, 0.5, 0), Vector(0, 0.5, 0.5), insideMaterial=myMaterial)]

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(myMaterial, left[0].insideMaterial)
        self.assertEqual(Vector(1, 0, 0), left[0].normal)
        self.assertEqual(Vector(1, 0, 0), right[0].normal)
        self.assertEqual(Vector(1, 0, 0), right[1].normal)

        self.assertEqual(len(right), 2)

    def test_givenAPolygonAndAPlane_whenSplittingOnAVertexInside_shouldReturn2Polygons(self):
        toBeSplit = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1))]
        splitValue = 0
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitTriangles(normal, dot)
        expectedLeft = [Triangle(Vector(1, -1, 1), Vector(1, 0, 1), Vector(0, 0, 0))]
        expectedRight = [Triangle(Vector(1, 1, 1), Vector(1, 0, 1), Vector(0, 0, 0))]
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right[0], expectedRight[0])

    def test_givenAPolygonAndAPlane_whenSplittingOnAVertexOutside_shouldReturn1Polygons(self):
        toBeSplit = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1))]
        splitValue = 1
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitTriangles(normal, dot)
        expectedLeft = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1))]
        expectedRight = []
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right, expectedRight)

    def test_givenAPolygonAndAPlane_whenSplittingOn2VerticesOutside_shouldReturn1Polygons(self):
        toBeSplit = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1))]
        splitValue = 1
        splitAxis = "x"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitTriangles(normal, dot)
        expectedLeft = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1))]
        expectedRight = []
        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right, expectedRight)

    def test_givenUltraThinPolygon_whenSplitting_shouldStillReturn2Polygons(self):
        vertices = [Vector(8.860660171779822, 5.000000000000001, -4.9455),
                    Vector(8.856599089933916, 4.995938918154095, -4.9455),
                    Vector(8.856599089933916, 4.99986899735981, -4.9455)]
        toBeSplit = [Triangle(*vertices)]
        splitAxis = "y"
        splitValue = 4.999868997359811
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)
        left, right = self._fbtc._splitTriangles(normal, dot)
        self.assertEqual(1, len(left))
        self.assertEqual(1, len(right))

    def test_givenANodeWith2Polygon_whenSplitting_shouldSplitBetweenPolygons(self):
        """This type of test is extremely sensitive on initial parameters."""
        expectedLeft = [Triangle(Vector(0, 0, 0), Vector(1, 1, 1), Vector(1, -1, 1)),
                        Triangle(Vector(0, 0, 0), Vector(-1, -1, -1), Vector(-2, -2, -3))]
        expectedRight = [Triangle(Vector(2, 4, 4), Vector(2, 2, 2), Vector(2, 2, 3)),
                         Triangle(Vector(2, 5, 5), Vector(2, 2, 2), Vector(2, 2, 3)),
                         Triangle(Vector(2, 6, 6), Vector(3, 3, 3), Vector(2, 2, 3))]
        polygons = []
        polygons.extend(expectedLeft)
        polygons.extend(expectedRight)
        nodeBbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        node = Node(polygons=polygons, bbox=nodeBbox)
        fbtc = SplitThreeAxesConstructor(traversalCost=8, intersectionCost=2)
        splitNodeResult = fbtc._splitNode(node)
        groups = splitNodeResult.polygonGroups
        self.assertEqual(2, len(groups))
        self.assertListEqual(expectedLeft, groups[0])
        self.assertListEqual(expectedRight, groups[1])
