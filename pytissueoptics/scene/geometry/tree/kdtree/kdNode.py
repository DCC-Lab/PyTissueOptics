from __future__ import annotations
from kd3ba.Axis import Axis


class KDNode:
    def __init__(self, parent=None, left=None, right=None, depth=0, axis=Axis("x"), triangles=None, boundingBox=None,
                 maxDepth=100, splitStrategy=None):
        self._parent = parent
        self._left = left
        self._right = right
        self._depth = depth
        self._axis = axis
        self._maxDepth = maxDepth
        self._triangles = triangles
        self._boundingBox = boundingBox
        self._splitStrategy = splitStrategy
        self.split()

    @property
    def isRoot(self):
        if self._parent is None:
            return True
        else:
            return False

    @property
    def triangles(self):
        return self._triangles

    @property
    def axis(self):
        return self._axis

    @property
    def boundingBox(self):
        return self._boundingBox

    def split(self):
        if self._depth < self._maxDepth and len(self._triangles) > 2:
            splitLine = self.calculateSplitLine()
            goingLeft, goingRight = self.countLeftRight(splitLine)
            self._left = KDNode(parent=self, triangles=goingLeft,
                                boundingBox=self._boundingBox.updateFrom(self._axis.axis, "max", splitLine),
                                axis=self._axis.nextAxisRotate(), depth=self._depth + 1, maxdepth=self._maxdepth,
                                splitStrategy=self._splitStrategy)
            self._right = KDNode(parent=self, triangles=goingRight,
                                 boundingBox=self._boundingBox.updateFrom(self._axis.axis, "min", splitLine),
                                 axis=self._axis.nextAxisRotate(), depth=self._depth + 1, maxdepth=self._maxdepth,
                                 splitStrategy=self._splitStrategy)

    def countLeftRight(self, line=None):
        goingLeft = []
        goingRight = []

        for triangle in self._triangles:
            if triangle.globalBoundingBox[self._axis.axis]["min"] < line and \
                    triangle.globalBoundingBox[self._axis.axis]["max"] < line:
                goingLeft.append(triangle)
            elif triangle.globalBoundingBox[self._axis.axis]["min"] > line and \
                    triangle.globalBoundingBox[self._axis.axis]["max"] > line:
                goingRight.append(triangle)
            else:
                goingLeft.append(triangle)
                goingRight.append(triangle)

        return goingLeft, goingRight

    def calculateSplitLine(self):
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
