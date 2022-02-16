from typing import List, Tuple
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.geometry.tree.splitUtils import AxisSelector, NodeSplitter, PolyCounter


class TreeStrategy:
    def __init__(self):
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None
        self._polygons = None
        self._nodeAxis = None
        self._nodeBbox = None
        self._splitAxis = "x"
        self._goingLeft = []
        self._goingRight = []

    def run(self, polygons: List[Polygon], nodeAxis: str, nodeBbox: BoundingBox)-> Tuple[str, float, List[Polygon], List[Polygon]]:
        self._polygons = polygons
        self._nodeAxis = nodeAxis
        self._nodeBbox = nodeBbox
        self._goingLeft = []
        self._goingRight = []
        self._selectSplitAxis()
        self._split()

    def _split(self) -> Tuple[str, List[Polygon], List[Polygon]]:
        raise NotImplementedError

    def _selectSplitAxis(self) -> str:
        self._splitAxis = self._axisSelector.run(self._nodeAxis)

    def _countLeftRight(self, line: float) -> Tuple:
        self._polyCounter.run()



