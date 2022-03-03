import unittest

from mockito import mock, verifyNoUnwantedInteractions, when, expect

from pytissueoptics.scene.geometry import Polygon, BoundingBox, Vector
from pytissueoptics.scene.tree import Node, SpacePartition
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, AxisSelector, PolygonCounter, NodeSplitter, \
    SplitNodeResult


class TestSpacePartition(unittest.TestCase):
    def setUp(self):
        poly = Polygon([Vector(0, 0, 0), Vector(1, 1, 0), Vector(1, 2, 3)])
        polyList = [poly, poly, poly, poly]
        bbox1 = BoundingBox(xLim=[0, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        bbox2 = BoundingBox(xLim=[0.5, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        bbox3 = BoundingBox(xLim=[0, 0.5], yLim=[0.5, 1], zLim=[0.5, 1])
        result1 = SplitNodeResult(False, "x", 0, [bbox1], [polyList])
        result2 = SplitNodeResult(False, "x", 0.5, [bbox2, bbox3], [polyList])
        result3 = SplitNodeResult(True, "x", 1, [bbox3], [polyList])

        self.AXIS_SELECTOR = mock(AxisSelector)
        when(self.AXIS_SELECTOR).select(...).thenReturn("x")

        self.POLY_COUNTER = mock(PolygonCounter)
        self.NODE_SPLITTER = mock(NodeSplitter)
        when(self.NODE_SPLITTER).setContext(...).thenReturn()
        expect(self.NODE_SPLITTER, times=3).split(...).thenReturn(result1).thenReturn(result2).thenReturn(result3)

        self.root = Node(polygons=polyList, bbox=bbox1)

        self.treeConstructor = TreeConstructor()
        self.treeConstructor.setContext(self.AXIS_SELECTOR, self.POLY_COUNTER, self.NODE_SPLITTER)
        self.tree = SpacePartition(bbox1, polyList, self.treeConstructor, minLeafSize=1)

    def testGetNodeCount_shouldReturnCountOf3(self):
        count = self.tree.getNodeCount()
        verifyNoUnwantedInteractions()
        self.assertEqual(3, count)

    def testGetLeafCount_shouldReturnCountOf1(self):
        count = self.tree.getLeafCount()
        verifyNoUnwantedInteractions()
        self.assertEqual(1, count)

    def testGetLeafBbox_shouldReturnBBox(self):
        bbox = BoundingBox(xLim=[0.5, 1], yLim=[0.5, 1], zLim=[0.5, 1])
        leafBboxes = self.tree.getLeafBoundingBoxes()
        verifyNoUnwantedInteractions()
        self.assertEqual([bbox], leafBboxes)

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
