import unittest

from mockito import mock, verify, when, patch

from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, AxisSelector, PolyCounter, NodeSplitter, \
    SplitNodeResult


class TestTreeConstructor(unittest.TestCase):
    def setUp(self):
        self.treeConstructor = TreeConstructor()

        patch(Node.__init__, )

        self.NODE = mock(Node)
        when(self.NODE)
        when(self.NODE).depth(...).thenReturn(1)
        when(self.NODE).maxDepth(...).thenReturn(2)
        when(self.NODE).maxLeafSize(...).thenReturn(3)
        when(self.NODE).polygons(...).thenReturn([1, 2])
        when(self.NODE).bbox(...).thenReturn()

        self.AXIS_SELECTOR = mock(AxisSelector)
        when(self.AXIS_SELECTOR).run(...).thenReturn("x")

        self.POLY_COUNTER = mock(PolyCounter)
        self.NODE_SPLITTER = mock(NodeSplitter)
        when(self.NODE_SPLITTER).run(...).thenReturn(SplitNodeResult(False, "x", 1, [1, 1], [1, 1]))

    def testGivenANode_whenSplitting_shouldReturnSplitNodeResult(self):
        pass

    def testGivenNodeToSplitTwice_shouldRecursiveCallTwice(self):
        self.treeConstructor.growTree(self.NODE)
        verify(self.AXIS_SELECTOR).run(...)
        verify(self.NODE_SPLITTER).run(...)


class TestScene(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()

    def testWhenAddingASolidAtAPosition_shouldPlaceTheSolidAtTheDesiredPosition(self):
        SOLID_POSITION = Vector(4, 0, 1)
        SOLID = mock(Solid)
        when(SOLID).translateTo(...).thenReturn()

        self.scene.add(SOLID, position=SOLID_POSITION)

        verify(SOLID).translateTo(SOLID_POSITION)

    def testWhenAddingASolidAtNoSpecificPosition_shouldKeepTheSolidAtItsPredefinedPosition(self):
        SOLID = mock(Solid)
        when(SOLID).translateTo(...).thenReturn()

        self.scene.add(SOLID)

        verify(SOLID, times=0).translateTo(...)

    def testWhenAddingASolidThatPartlyOverlapsWithAnotherOne_shouldNotAdd(self):
        OTHER_SOLID = self.makeSolidWith(BoundingBox([2, 5], [2, 5], [2, 5]))
        SOLID = self.makeSolidWith(BoundingBox([0, 3], [0, 3], [0, 3]))
        self.scene.add(OTHER_SOLID)

        with self.assertRaises(Exception):
            self.scene.add(SOLID)
