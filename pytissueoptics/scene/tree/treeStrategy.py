from typing import List, Tuple
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.binary.splitUtils import PolyCounter, AxisSelector, NodeSplitter
from pytissueoptics.scene.tree.splitNodeResult import SplitNodeResult


class TreeStrategy:
    def __init__(self):
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None
        self._splitNodeResult: SplitNodeResult = None
        self._polygons = None
        self._nodeDepth = None
        self._nodeBbox = None

        self._loadComponents()

    def _loadComponents(self):
        raise NotImplementedError

    def run(self, polygons: List[Polygon], nodeDepth: int, nodeBbox: BoundingBox)-> SplitNodeResult:
        self._polygons = polygons
        self._nodeDepth = nodeDepth
        self._nodeBbox = nodeBbox
        self._selectSplitAxis()
        self._split()
        return self._splitNodeResult


    def _split(self):
        self._splitNodeResult = self._nodeSplitter.run(self._splitAxis, self._polygons, self._nodeBbox)

    def _selectSplitAxis(self):
        self._splitAxis = self._axisSelector.run(self._nodeDepth, self._polygons)
