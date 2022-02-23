import unittest

from mockito import mock, verifyNoUnwantedInteractions, when, expect

from pytissueoptics.scene.geometry import Polygon, BoundingBox, Vector
from pytissueoptics.scene.tree import Node, Tree
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, AxisSelector, PolyCounter, NodeSplitter, \
    SplitNodeResult


class TestTree(unittest.TestCase):
    def setUp(self):
        poly = Polygon([Vector(0, 0, 0), Vector(1, 1, 0), Vector(1, 2, 3)])
        polyList = [poly, poly, poly, poly]
        bbox = BoundingBox(xLim=[0, 1], yLim=[0, 1], zLim=[0, 1])
        result1 = SplitNodeResult(False, "x", 1, [bbox], [polyList])
        result2 = SplitNodeResult(False, "x", 1, [bbox], [polyList])
        result3 = SplitNodeResult(True, "x", 1, [bbox], [polyList])

        self.AXIS_SELECTOR = mock(AxisSelector)
        when(self.AXIS_SELECTOR).run(...).thenReturn("x")

        self.POLY_COUNTER = mock(PolyCounter)
        self.NODE_SPLITTER = mock(NodeSplitter)
        expect(self.NODE_SPLITTER, times=3).run(...).thenReturn(result1).thenReturn(result2).thenReturn(result3)

        self.root = Node(polygons=polyList, bbox=bbox, maxLeafSize=1)

        self.treeConstructor = TreeConstructor()
        self.treeConstructor.setContext(self.AXIS_SELECTOR, self.POLY_COUNTER, self.NODE_SPLITTER)
        self.tree = Tree(bbox, polyList, self.treeConstructor, maxLeafSize=1)

    def testGetNodeCount_shouldReturnCountOf3(self):
        count = self.tree.getNodeCount()
        self.assertEqual(3, count)

    def testGetLeafCount_shouldReturnCountOf1(self):
        count = self.tree.getLeafCount()
        self.assertEqual(1, count)

    def testGetLeafCount_shouldReturnCountOf1(self):
        count = self.tree.getLeafCount()
        self.assertEqual(1, count)
