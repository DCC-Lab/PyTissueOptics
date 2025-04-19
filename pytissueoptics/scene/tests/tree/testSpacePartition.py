import unittest

from mockito import expect, verifyNoUnwantedInteractions

from pytissueoptics.scene.geometry import BoundingBox, Polygon, Vector, Vertex
from pytissueoptics.scene.tree import Node, SpacePartition
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult, TreeConstructor


class TestSpacePartition(unittest.TestCase):
    def setUp(self):
        poly = Polygon([Vertex(0, 0, 0), Vertex(1, 1, 0), Vertex(1, 2, 3)])
        self.polyList = [poly, poly, poly, poly]
        bbox1 = BoundingBox(xLim=[0, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        bbox2 = BoundingBox(xLim=[0.5, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        bbox3 = BoundingBox(xLim=[0, 0.5], yLim=[0.5, 1], zLim=[0.5, 1])
        result1 = SplitNodeResult(False, [bbox1], [self.polyList])
        result2 = SplitNodeResult(False, [bbox2, bbox3], [self.polyList])
        result3 = SplitNodeResult(True, [bbox3], [self.polyList])

        self.root = Node(polygons=self.polyList, bbox=bbox1)
        self.treeConstructor = TreeConstructor()
        expect(self.treeConstructor, times=3)._splitNode(...).thenReturn(result1).thenReturn(result2).thenReturn(result3)
        self.tree = SpacePartition(bbox1, self.polyList, self.treeConstructor, minLeafSize=1)

    def testShouldHaveNodeCount(self):
        count = self.tree.getNodeCount()
        verifyNoUnwantedInteractions()
        self.assertEqual(3, count)

    def testShouldHaveLeafCount(self):
        count = self.tree.getLeafCount()
        verifyNoUnwantedInteractions()
        self.assertEqual(1, count)

    def testShouldHaveLeafBBoxes(self):
        bbox = BoundingBox(xLim=[0.5, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        leafBboxes = self.tree.getLeafBoundingBoxes()
        verifyNoUnwantedInteractions()
        self.assertEqual([bbox], leafBboxes)

    def testShouldHaveMaxDepth(self):
        depth = self.tree.getMaxDepth()
        verifyNoUnwantedInteractions()
        self.assertEqual(2, depth)

    def testShouldHaveAverageDepth(self):
        avgDepth = self.tree.getAverageDepth()
        verifyNoUnwantedInteractions()
        self.assertEqual(2, avgDepth)

    def testShouldHaveAverageLeafSize(self):
        avgLeafSize = self.tree.getAverageLeafSize()
        verifyNoUnwantedInteractions()
        self.assertEqual(4, avgLeafSize)

    def testShouldHaveLeafPolygons(self):
        leafPolygons = self.tree.getLeafPolygons()
        verifyNoUnwantedInteractions()
        self.assertEqual(self.polyList, leafPolygons)

    def testGivenAPointOnlyInLeafNode_whenSearchPoint_shouldReturnLeafNode(self):
        point = Vector(0.6, 0.6, 0.6)
        node = self.tree.searchPoint(point)
        expectedNodeBbox = BoundingBox(xLim=[0.5, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        self.assertEqual(expectedNodeBbox, node.bbox)

    def testGivenPointNotInLeaf_whenSearchPoint_shouldReturnCorrectNode(self):
        point = Vector(0.4, 0.6, 0.6)
        node = self.tree.searchPoint(point)
        expectedNodeBbox = BoundingBox(xLim=[0, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        self.assertEqual(expectedNodeBbox, node.bbox)

    def testGivenOutsideVector_whenSearchPoint_shouldReturnNone(self):
        point = Vector(0.1, 0.4, 0.6)
        node = self.tree.searchPoint(point)
        expectedNodeBbox = None
        self.assertEqual(expectedNodeBbox, node)
