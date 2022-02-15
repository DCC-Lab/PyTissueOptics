from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.geometry.tree.kdtree.splitter import Splitter
from pytissueoptics.scene.scene import Scene


class KDNode:
    def __init__(self, parent: 'KDNode' = None, leftNode: 'KDNode' = None, rightNode: 'KDNode' = None, depth: int = 0,
                 axis: str = "x", polygons: List[Polygon] = None, boundingBox: BoundingBox = None, scene: Scene = None,
                 maxDepth=100, splitStrategy: Splitter = None):
        self._parent = parent
        self._leftNode = leftNode
        self._rightNode = rightNode
        self._depth = depth
        self._axis = axis
        self._maxDepth = maxDepth
        self._scene = scene
        self._polygons = polygons
        self._boundingBox = boundingBox
        self._splitStrategy = splitStrategy

        if self.isRoot:
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
        return self._axis

    @property
    def boundingBox(self) -> BoundingBox:
        return self._boundingBox

    def split(self):
        if self._depth < self._maxDepth and len(self._polygons) > 2:
            splitLine = self._calculateSplitLine()
            goingLeft, goingRight = self._separateLeftRight(splitLine)
            self._leftNode = KDNode(parent=self, polygons=goingLeft,
                                    boundingBox=self._boundingBox.changeToNew(self._axis, "max", splitLine),
                                    axis=self._nextAxis(), depth=self._depth + 1, maxDepth=self._maxDepth,
                                    splitStrategy=self._splitStrategy)

            self._rightNode = KDNode(parent=self, polygons=goingRight,
                                     boundingBox=self._boundingBox.changeToNew(self._axis, "min", splitLine),
                                     axis=self._nextAxis(), depth=self._depth + 1, maxDepth=self._maxDepth,
                                     splitStrategy=self._splitStrategy)

    def _separateLeftRight(self, line: float) -> Tuple:
        goingLeft = []
        goingRight = []

        for polygon in self._polygons:
            if polygon.bbox.getAxisLimit(self._axis, "min") < line and \
                    polygon.bbox.getAxisLimit(self._axis, "max") < line:
                goingLeft.append(polygon)
            elif polygon.bbox.getAxisLimit(self._axis, "min") > line and \
                    polygon.bbox.getAxisLimit(self._axis, "max") > line:
                goingRight.append(polygon)
            else:
                goingLeft.append(polygon)
                goingRight.append(polygon)

        return goingLeft, goingRight

    def _calculateSplitLine(self):
        return self._splitStrategy.calculateSplitLine(self._polygons, self._axis)

    def _nextAxis(self) -> str:
        if self._axis == "x":
            return "y"
        elif self._axis == "y":
            return "z"
        elif self._axis == "z":
            return "x"

    @staticmethod
    def getBoundingBoxes(node: 'KDNode', bboxList: List) -> List[BoundingBox]:
        if bboxList is None:
            bboxList = []

        if node is not None:
            bboxList.append(node.boundingBox)
            node.getBoundingBoxes(node.leftNode, bboxList)
            node.getBoundingBoxes(node.rightNode, bboxList)

            if node.isRoot:
                return bboxList
