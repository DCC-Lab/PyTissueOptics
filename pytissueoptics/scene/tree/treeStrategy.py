from pytissueoptics.scene.tree.splitUtils import PolyCounter, AxisSelector, NodeSplitter
from pytissueoptics.scene.tree import SplitNodeResult


class TreeStrategy:
    def __init__(self):
        self._splitNodeResult: SplitNodeResult = None
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None
        self._polygons = None
        self._nodeDepth = None
        self._nodeBbox = None
        self._splitAxis = None
        self._loadComponents()

    def _loadComponents(self):
        raise NotImplementedError

    def run(self, bbox, polygons, depth)-> SplitNodeResult:
        self._polygons = polygons
        self._nodeDepth = depth
        self._nodeBbox = bbox
        self._selectSplitAxis()
        self._split()
        return self._splitNodeResult

    def _split(self):
        self._splitNodeResult = self._nodeSplitter.run(self._splitAxis, self._polygons, self._nodeBbox)

    def _selectSplitAxis(self):
        self._splitAxis = self._axisSelector.run(self._nodeDepth, self._nodeBbox, self._polygons)
