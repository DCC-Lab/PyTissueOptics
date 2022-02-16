from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.binary import BinaryTreeStrategy
from pytissueoptics.scene.scene import Scene


class BinaryNode:
    def __init__(self, parent: 'BinaryNode' = None, leftNode: 'BinaryNode' = None, rightNode: 'BinaryNode' = None, depth: int = 0,
                 axis: str = "x", polygons: List[Polygon] = None, boundingBox: BoundingBox = None, scene: Scene = None,
                 maxDepth=100, treeStrategy: BinaryTreeStrategy = None):
        self._parent = parent
        self._leftNode = leftNode
        self._rightNode = rightNode
        self._depth = depth
        self._splitAxis = axis
        self._maxDepth = maxDepth
        self._scene = scene
        self._polygons = polygons
        self._boundingBox = boundingBox
        self._treeStrategy = treeStrategy

        if self.isRoot and scene is not None:
            self._polygons = self._scene.getPolygons()
            self._boundingBox = self._scene.getBoundingBox()

        self.split()

    @property
    def parent(self):
        return self._parent

    @property
    def isRoot(self):
        if self._parent is None:
            return True
        else:
            return False

    @property
    def isLeaf(self):
        if self._leftNode is None and self._rightNode is None:
            return True
        else:
            return False

    @property
    def leftNode(self):
        return self._leftNode

    @property
    def rightNode(self):
        return self._rightNode

    @property
    def polygons(self) -> List[Polygon]:
        return self._polygons

    @property
    def axis(self) -> str:
        return self._splitAxis

    @property
    def boundingBox(self) -> BoundingBox:
        return self._boundingBox

    def split(self):
        if self._depth < self._maxDepth and len(self._polygons) > 2:
            newSplitAxis, splitLine, goingLeft, goingRight = self._split()
            self._splitAxis = newSplitAxis
            if len(goingLeft) != len(self._polygons):
                self._leftNode = BinaryNode(parent=self, polygons=goingLeft,
                                            boundingBox=self._boundingBox.changeToNew(newSplitAxis, "max", splitLine),
                                            axis=newSplitAxis, depth=self._depth + 1, maxDepth=self._maxDepth,
                                            treeStrategy=self._treeStrategy)
            if len(goingRight) != len(self._polygons):
                self._rightNode = BinaryNode(parent=self, polygons=goingRight,
                                             boundingBox=self._boundingBox.changeToNew(newSplitAxis, "min", splitLine),
                                             axis=newSplitAxis, depth=self._depth + 1, maxDepth=self._maxDepth,
                                             treeStrategy=self._treeStrategy)

    def _split(self):
        return self._treeStrategy.run(self._polygons, self._splitAxis, self._boundingBox)


    @staticmethod
    def getLeafBoundingBoxes(node: 'BinaryNode', bboxList: List) -> List[BoundingBox]:
        if bboxList is None:
            bboxList = []

        if node is not None:
            if not node.isLeaf:
                node.getLeafBoundingBoxes(node.leftNode, bboxList)
                node.getLeafBoundingBoxes(node.rightNode, bboxList)

            else:
                bboxList.append(node.boundingBox)

            if node.isRoot:
                return bboxList
