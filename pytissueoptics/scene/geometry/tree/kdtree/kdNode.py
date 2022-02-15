from typing import Tuple, List
from pytissueoptics.scene.geometry import Vector, Polygon, BoundingBox
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.scene import Scene


class KDNode:
    def __init__(self, parent: 'KDNode' = None, left: 'KDNode' = None, right: 'KDNode' = None, depth: int = 0,
                 axis: str = "x", scene: Scene = None, maxDepth=100, splitStrategy=None):
        self._parent = parent
        self._left = left
        self._right = right
        self._depth = depth
        self._axis = axis
        self._maxDepth = maxDepth
        self._scene = scene
        self._polygons = None
        self._boundingBox = None
        self._splitStrategy = splitStrategy



        self.split()

    @property
    def isRoot(self):
        if self._parent is None:
            return True
        else:
            return False

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
            goingLeft, goingRight = self._countLeftRight(splitLine)
            self._left = KDNode(parent=self, polygons=goingLeft,
                                boundingBox=self._boundingBox.updateFrom(self._axis.axis, "max", splitLine),
                                axis=self._axis.nextAxisRotate(), depth=self._depth + 1, maxdepth=self._maxdepth,
                                splitStrategy=self._splitStrategy)
            self._right = KDNode(parent=self, polygons=goingRight,
                                 boundingBox=self._boundingBox.updateFrom(self._axis.axis, "min", splitLine),
                                 axis=self._axis.nextAxisRotate(), depth=self._depth + 1, maxdepth=self._maxdepth,
                                 splitStrategy=self._splitStrategy)

    def _countLeftRight(self, line: float) -> Tuple:
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
        return self._splitStrategy.calculateSplitLine(self)

    def getAllBBox(self, node, bboxList=None):
        if bboxList is None:
            bboxList = []

        if node is not None:
            bboxList.append(node._boundingBox)
            node.getAllBBox(node._left, bboxList)
            node.getAllBBox(node._right, bboxList)

            if node.isRoot:
                return bboxList

    @staticmethod
    def changeAxis(currentAxis: str) -> str:
        if currentAxis == "x":
            return "y"
        elif currentAxis == "y":
            return "z"
        elif currentAxis == "z":
            return "x"
