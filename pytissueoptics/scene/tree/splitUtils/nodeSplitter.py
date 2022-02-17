from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree import SplitNodeResult


class NodeSplitter:
    def __init__(self, polyCounter):
        self._splitAxis = None
        self._polygons = None
        self._nodeBbox = None
        self._splitNodeResult = None
        self._polyCounter = polyCounter

    def run(self, splitAxis: str, polygons: List[Polygon], nodeBbox: BoundingBox) -> SplitNodeResult:
        self._splitAxis = splitAxis
        self._polygons = polygons
        self._nodeBbox = nodeBbox
        self._run()
        return self._splitNodeResult

    def _run(self):
        raise NotImplementedError
