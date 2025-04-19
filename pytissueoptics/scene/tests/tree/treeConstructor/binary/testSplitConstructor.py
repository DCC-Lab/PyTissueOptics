import unittest
from math import sqrt

from pytissueoptics.scene.geometry import BoundingBox, Environment, Triangle, Vector, Vertex
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor.binary import SAHSearchResult, SplitThreeAxesConstructor


class TestSplitConstructor(unittest.TestCase):
    def setUp(self) -> None:
        self._fbtc = SplitThreeAxesConstructor()

    def testGivenATriangleAndAPlane_whenPolygonAsRays_shouldReturnCorrectRays(self):
        triangle = Triangle(Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(1, 1, 0))
        triangleRays = self._fbtc._getPolygonAsRays(triangle)
        expectedTriangleRays = [Ray(Vector(0, 0, 0), Vector(1, 0, 0), 1.0),
                                Ray(Vector(1, 0, 0), Vector(0, 1, 0), 1.0),
                                Ray(Vector(1, 1, 0), Vector(-sqrt(2) / 2, -sqrt(2) / 2, 0), sqrt(2))]

        for i, tri in enumerate(triangleRays):
            self.assertEqual(tri.direction, expectedTriangleRays[i].direction, 2)
            self.assertEqual(tri.origin, expectedTriangleRays[i].origin, 2)
            self.assertEqual(tri.length, expectedTriangleRays[i].length, 2)

    def testGivenAPolygonAndAPlane_whenSplittingPolygon_shouldReturn3Polygons(self):
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        splitValue = 0.5
        splitAxis = "x"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedLeft = Triangle(Vertex(0, 0, 0), Vertex(0.5, 0.5, 0.5), Vertex(0.5, -0.5, 0.5))
        expectedRight1 = Triangle(Vertex(0.5, 0.5, 0.5), Vertex(1, 1, 1), Vertex(0.5, -0.5, 0.5))
        expectedRight2 = Triangle(Vertex(0.5, -0.5, 0.5), Vertex(1, 1, 1), Vertex(1, -1, 1))

        self.assertEqual(1, len(left))
        self.assertEqual(2, len(right))
        self.assertIn(expectedLeft, left)
        self.assertIn(expectedRight1, right)
        self.assertIn(expectedRight2, right)

    def testGivenAPolygonAndAPlane_whenSplittingPolygon_splitPolygonShouldHaveGoodEnvironmentAndNormal(self):
        myEnvironment = Environment("A material")
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(0, 1, 0), Vertex(0, 1, 1), insideEnvironment=myEnvironment)]
        splitValue = 0.5
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedLeft = [Triangle(Vertex(0, 0, 0), Vertex(0, 0.5, 0), Vertex(0, 0.5, 0.5), insideEnvironment=myEnvironment)]

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(myEnvironment, left[0].insideEnvironment)
        self.assertEqual(Vector(1, 0, 0), left[0].normal)
        self.assertEqual(Vector(1, 0, 0), right[0].normal)
        self.assertEqual(Vector(1, 0, 0), right[1].normal)
        self.assertEqual(len(right), 2)

    def testGivenAPolygonAndAPlane_whenSplittingOnAVertexInside_shouldReturn2Polygons(self):
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        splitValue = 0
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedLeft = [Triangle(Vertex(1, -1, 1), Vertex(1, 0, 1), Vertex(0, 0, 0))]
        expectedRight = [Triangle(Vertex(1, 1, 1), Vertex(1, 0, 1), Vertex(0, 0, 0))]

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right[0], expectedRight[0])

    def testGivenAPolygonAndAPlane_whenBarelySplittingOnAVertexInside_shouldReturn2Polygons(self):
        tolerance = 0.000001
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -tolerance, 1))]
        splitValue = 0
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedLeft = [Triangle(Vertex(1, -tolerance, 1), Vertex(1, 0, 1), Vertex(0, 0, 0))]
        expectedRight = [Triangle(Vertex(1, 1, 1), Vertex(1, 0, 1), Vertex(0, 0, 0))]

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right[0], expectedRight[0])

    def testGivenAPolygonAndAPlane_whenBarelySplittingBetweenTwoVertices_shouldReturn3Polygons(self):
        tolerance = 0.000001
        toBeSplit = [Triangle(Vertex(0, tolerance, 0), Vertex(1, 1, 0), Vertex(1, -tolerance, 0))]
        splitValue = 0
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        self.assertEqual(1, len(left))
        self.assertEqual(2, len(right))

    def testGivenAPolygonAndAPlane_whenBarelySplittingOnTwoVertices_shouldNotSplit(self):
        tolerance = 0.000001/2
        toBeSplit = [Triangle(Vertex(0, tolerance, 0), Vertex(1, 1, 0), Vertex(1, -tolerance, 0))]
        splitValue = 0
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        self.assertEqual(0, len(left))
        self.assertEqual(1, len(right))
        self.assertEqual(toBeSplit[0], right[0])

    def testGivenAPolygonAndAPlane_whenSplittingOnAVertexOutsideRight_shouldReturn1PolygonLeft(self):
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        splitValue = 1
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedLeft = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        expectedRight = []

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right, expectedRight)

    def testGivenAPolygonAndAPlane_whenSplittingOnAVertexOutsideLeft_shouldReturn1PolygonRight(self):
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        splitValue = -1
        splitAxis = "y"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedRight = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        expectedLeft = []

        self.assertEqual(left, expectedLeft)
        self.assertEqual(right[0], expectedRight[0])

    def testGivenAPolygonAndAPlane_whenSplittingOn2VerticesOutside_shouldReturn1Polygons(self):
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        splitValue = 1
        splitAxis = "x"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        expectedLeft = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1))]
        expectedRight = []

        self.assertEqual(left[0], expectedLeft[0])
        self.assertEqual(right, expectedRight)

    def testGivenAPolygonAndAPlane_whenSplittingOnPolygonPlane_shouldNotSplitAndReturnInBothLeftAndRight(self):
        toBeSplit = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 0), Vertex(1, -1, 0))]
        splitValue = 0
        splitAxis = "z"
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        self.assertEqual(1, len(left))
        self.assertEqual(1, len(right))
        self.assertEqual(left[0], toBeSplit[0])
        self.assertEqual(right[0], toBeSplit[0])

    def testGivenUltraThinPolygon_whenSplitting_shouldStillReturn2Polygons(self):
        vertices = [Vertex(8.860660171779822, 5.000000000000001, -4.9455),
                    Vertex(8.856599089933916, 4.995938918154095, -4.9455),
                    Vertex(8.856599089933916, 4.99986899735981, -4.9455)]
        toBeSplit = [Triangle(*vertices)]
        splitAxis = "y"
        splitValue = 4.999868997359811
        self._fbtc.result = SAHSearchResult([], [], toBeSplit, None, None, splitAxis, splitValue)
        normal, dot = self._fbtc._makeSplitPlane(splitAxis, splitValue)

        left, right = self._fbtc._splitTriangles(normal, dot)

        self.assertEqual(1, len(left))
        self.assertEqual(1, len(right))

    def testGivenANodeWith2Polygon_whenSplitting_shouldSplitBetweenPolygons(self):
        """This type of test is extremely sensitive on initial parameters."""
        expectedLeft = [Triangle(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(1, -1, 1)),
                        Triangle(Vertex(0, 0, 0), Vertex(-1, -1, -1), Vertex(-2, -2, -3))]
        expectedRight = [Triangle(Vertex(2, 4, 4), Vertex(2, 2, 2), Vertex(2, 2, 3)),
                         Triangle(Vertex(2, 5, 5), Vertex(2, 2, 2), Vertex(2, 2, 3)),
                         Triangle(Vertex(2, 6, 6), Vertex(3, 3, 3), Vertex(2, 2, 3))]
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
