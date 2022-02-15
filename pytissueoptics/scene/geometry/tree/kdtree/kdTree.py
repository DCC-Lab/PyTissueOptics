from typing import List
from kd3ba.KDNode import KDNode
from kd3ba.Polygon import Triangle
from kd3ba.BoundingBox import BoundingBox
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


class KDTree:
    def __init__(self, maxBoundingBox: BoundingBox, objectsList: List[Triangle], maxDepth=100, splitStrategy=None):
        self._maxBoundingBox = maxBoundingBox
        self._objectsList = objectsList
        self._maxDepth = maxDepth
        self._splitStrategy = splitStrategy
        self._tree = KDNode(triangles=objectsList, maxDepth=self._maxDepth, boundingBox=self._maxBoundingBox,
                            splitStrategy=self._splitStrategy)

    def getAllBBox(self):
        bboxList = self._tree.getAllBBox(self._tree)
        return bboxList

    def drawTree(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_box_aspect((1, 1, 1))

        # draw kdTree as cube lines
        for bbox in self.getAllBBox():
            x = [bbox.xLim.min, bbox.xLim.max]
            y = [bbox.yLim.min, bbox.yLim.max]
            z = [bbox.zLim.min, bbox.zLim.max]

            allPossibleBoundaryPoints = np.array(list(product(x, y, z)))
            allPossibleBoundaryLines = combinations(allPossibleBoundaryPoints, 2)
            for i, (pointA, pointB) in enumerate(allPossibleBoundaryLines):
                difference = (pointA - pointB)
                nonZeroCoordinateDifference = list(filter(lambda a: a != 0, difference))
                if len(nonZeroCoordinateDifference) == 1:
                    ax.plot3D(*zip(pointA, pointB), color="k", linewidth=0.25)
