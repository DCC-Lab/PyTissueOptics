from typing import List

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult


class NodeSplitter:
    def __init__(self, polyCounter, **kwargs):
        self._polyCounter = polyCounter
        self.kwargs = kwargs

    def run(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        raise NotImplementedError

    @staticmethod
    def _getNewChildrenBbox(nodeBbox, splitAxis, splitLine) -> List[BoundingBox]:
        leftBbox = nodeBbox.changeToNew(splitAxis, "max", splitLine)
        rightBbox = nodeBbox.changeToNew(splitAxis, "min", splitLine)
        return [leftBbox, rightBbox]
