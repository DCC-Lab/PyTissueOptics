import unittest

from mockito import mock, verifyNoUnwantedInteractions, when, expect

from pytissueoptics.scene.geometry import Polygon, BoundingBox, Vector
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, AxisSelector, PolygonCounter, NodeSplitter, \
    SplitNodeResult


class TestTreeConstructor(unittest.TestCase):
    def setUp(self):
        poly = Polygon([Vector(0, 0, 0), Vector(1, 1, 0), Vector(1, 2, 3)])
        polyList = [poly, poly, poly, poly]
        bbox = BoundingBox(xLim=[0, 1], yLim=[0, 1], zLim=[0, 1])
        result1 = SplitNodeResult(False, "x", 1, [bbox], [polyList])
        result2 = SplitNodeResult(False, "x", 1, [bbox], [polyList])
        result3 = SplitNodeResult(True, "x", 1, [bbox], [polyList])

        self.AXIS_SELECTOR = mock(AxisSelector)
        when(self.AXIS_SELECTOR).select(...).thenReturn("x")

        self.POLY_COUNTER = mock(PolygonCounter)
        self.NODE_SPLITTER = mock(NodeSplitter)
        when(self.NODE_SPLITTER).setContext(...).thenReturn()
        expect(self.NODE_SPLITTER, times=3).split(...).thenReturn(result1).thenReturn(result2).thenReturn(result3)

        self.root = Node(polygons=polyList, bbox=bbox)

        self.treeConstructor = TreeConstructor()
        self.treeConstructor.setContext(self.AXIS_SELECTOR, self.POLY_COUNTER, self.NODE_SPLITTER)

    def testGrowTree_givenNodeToSplitTwice_shouldRecursiveCallThreeTimeAndMake2Child(self):
        self.treeConstructor.growTree(self.root, maxDepth=10, minLeafSize=1)
        self.assertEqual(1, len(self.root.children))
        self.assertEqual(1, len(self.root.children[0].children))
        self.assertEqual(0, len(self.root.children[0].children[0].children))
        verifyNoUnwantedInteractions()
