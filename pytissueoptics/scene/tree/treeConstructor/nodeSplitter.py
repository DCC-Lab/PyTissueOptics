from typing import List

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult


class NodeSplitter:
    def __init__(self, polyCounter, **kwargs):
        self._polyCounter = polyCounter
        self.kwargs = kwargs

    def run(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        raise NotImplementedError

    def _getNewChildrenBbox(self, nodeBbox, splitAxis, splitLine) -> List[BoundingBox]:
        leftBbox = self._copyBboxThenUpdate(nodeBbox, splitAxis, "max", splitLine)
        rightBbox = self._copyBboxThenUpdate(nodeBbox, splitAxis, "min", splitLine)
        return [leftBbox, rightBbox]

    @staticmethod
    def _copyBboxThenUpdate(bbox: BoundingBox, axis: str, limit: str, value: float) -> BoundingBox:
        newBbox = bbox.copy()
        newBbox.update(axis, limit, value)
        return newBbox
