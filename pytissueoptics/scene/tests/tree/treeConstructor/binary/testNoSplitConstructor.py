import unittest

from pytissueoptics.scene.geometry import Triangle, Vertex
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitOneAxisConstructor


class TestModernKDTreeConstructor(unittest.TestCase):
    def setUp(self) -> None:
        self._fbtc = NoSplitOneAxisConstructor()

    def testGivenPolygons_whenClassifying_shouldReturnCorrect3Groups(self):
        triangle0 = Triangle(Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(1, 1, 0))
        triangle1 = Triangle(Vertex(-5, -5, 0), Vertex(-5, 5, 0), Vertex(5, 5, 0))
        triangle2 = Triangle(Vertex(3, -3, 2), Vertex(1, -1, 1), Vertex(1, -1, 1))

        toClassify = [triangle1, triangle2, triangle0]
        self._fbtc.currentNode = Node(polygons=toClassify)
        left, right, both = self._fbtc._classifyNodePolygons("y", -0.5)
        self.assertListEqual(left, [triangle2])
        self.assertEqual(left[0].normal, triangle2.normal)
        self.assertListEqual(right, [triangle0])
        self.assertEqual(right[0].normal, triangle0.normal)
        self.assertListEqual(both, [triangle1])
        self.assertEqual(both[0].normal, triangle1.normal)

    def testGivenTrianglesWith1ContainedVertex_whenClassifying_shouldSendTriangleOnGoodSide(self):
        with self.subTest("left"):
            triangle0 = Triangle(Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(1, 1, 0))
            toClassify = [triangle0]
            self._fbtc.currentNode = Node(polygons=toClassify)
            left, right, both = self._fbtc._classifyNodePolygons("y", 1)
            self.assertListEqual(left, [triangle0])
            self.assertListEqual(right, [])
            self.assertListEqual(both, [])
        with self.subTest("right"):
            triangle0 = Triangle(Vertex(0, 2, 0), Vertex(1, 2, 0), Vertex(1, 1, 0))
            toClassify = [triangle0]
            self._fbtc.currentNode = Node(polygons=toClassify)
            left, right, both = self._fbtc._classifyNodePolygons("y", 1)
            self.assertListEqual(left, [])
            self.assertListEqual(right, [triangle0])
            self.assertListEqual(both, [])
        with self.subTest("toSplit"):
            triangle0 = Triangle(Vertex(0, 2, 0), Vertex(1, 0, 0), Vertex(1, 1, 0))
            toClassify = [triangle0]
            self._fbtc.currentNode = Node(polygons=toClassify)
            left, right, toSplit = self._fbtc._classifyNodePolygons("y", 1)
            self.assertListEqual(left, [])
            self.assertListEqual(right, [])
            self.assertListEqual(toSplit, [triangle0])

    def testGivenTrianglesWith2ContainedVertex_whenClassifying_shouldSendTriangleOnGoodSide(self):
        with self.subTest("left"):
            triangle0 = Triangle(Vertex(0, 0, 0), Vertex(1, 1, 0), Vertex(1, 1, 0))
            toClassify = [triangle0]
            self._fbtc.currentNode = Node(polygons=toClassify)
            left, right, both = self._fbtc._classifyNodePolygons("y", 1)
            self.assertListEqual(left, [triangle0])
            self.assertListEqual(right, [])
            self.assertListEqual(both, [])
        with self.subTest("right"):
            triangle0 = Triangle(Vertex(0, 2, 0), Vertex(1, 1, 0), Vertex(1, 1, 0))
            toClassify = [triangle0]
            self._fbtc.currentNode = Node(polygons=toClassify)
            left, right, both = self._fbtc._classifyNodePolygons("y", 1)
            self.assertListEqual(left, [])
            self.assertListEqual(right, [triangle0])
            self.assertListEqual(both, [])

    def testGivenTrianglesWith3ContainedVertex_whenClassifying_shouldSendLeftAndToSplit(self):
        triangle00 = Triangle(Vertex(0, 1, 0), Vertex(1, 0, 0), Vertex(1, 1, 0))
        triangle0 = Triangle(Vertex(0, 1, 0), Vertex(1, 1, 0), Vertex(1, 1, 0))
        toClassify = [triangle00, triangle0]
        self._fbtc.currentNode = Node(polygons=toClassify)
        left, right, toSplit = self._fbtc._classifyNodePolygons("y", 1)
        self.assertListEqual(left, [triangle00])
        self.assertListEqual(toSplit, [triangle0])
