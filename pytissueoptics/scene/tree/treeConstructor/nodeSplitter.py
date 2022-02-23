from typing import List

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult, PolygonCounter


class NodeSplitter:
    def __init__(self, polygonCounter: PolygonCounter = None, **kwargs):
        self._polygonCounter = polygonCounter
        self.kwargs = kwargs

    def setContext(self, polygonCounter: PolygonCounter, **kwargs):
        self._polygonCounter = polygonCounter
        self.kwargs = kwargs

    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        raise NotImplementedError

    @staticmethod
    def _getNewChildrenBbox(nodeBbox, splitAxis, splitLine) -> List[BoundingBox]:
        leftBbox = nodeBbox.copy()
        leftBbox.update(splitAxis, "max", splitLine)
        rightBbox = nodeBbox.copy()
        rightBbox.update(splitAxis, "min", splitLine)
        return [leftBbox, rightBbox]
