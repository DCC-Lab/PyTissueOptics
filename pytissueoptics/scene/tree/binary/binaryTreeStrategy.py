from typing import List, Tuple
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.geometry.tree.splitUtils import PolyCounter, AxisSelector, NodeSplitter


class TreeStrategy:
    def __init__(self):
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None
        self._polygons = None
        self._nodeAxis = None
        self._nodeBbox = None
        self._splitLine = None
        self._splitAxis = None
        self._goingLeft = []
        self._goingRight = []
        self._loadComponents()

    def _loadComponents(self):
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None

    def run(self,
            polygons: List[Polygon],
            nodeAxis: str,
            nodeBbox: BoundingBox)-> Tuple[str, float, List[Polygon], List[Polygon]]:

        self._polygons = polygons
        self._nodeAxis = nodeAxis
        self._nodeBbox = nodeBbox
        self._goingLeft = []
        self._goingRight = []
        self._selectSplitAxis()
        self._split()
        self._sendLeftRight()

        return self._splitAxis, self._splitLine, self._goingLeft, self._goingRight

    def _sendLeftRight(self):
        self._goingLeft, self._goingRight = self._polyCounter.run(self._splitLine, self._splitAxis, self._polygons)

    def _split(self):
        self._splitLine = self._nodeSplitter.run(self._splitAxis, self._polygons, self._nodeBbox)

    def _selectSplitAxis(self):
        self._splitAxis = self._axisSelector.run(self._nodeAxis, self._polygons)


class BasicKDTreeStrategy(TreeStrategy):
    def loadComponents(self):
        from pytissueoptics.scene.geometry.tree.splitUtils.nodeSplitter import CentroidNodeSplitter
        from pytissueoptics.scene.geometry.tree.splitUtils.polyCounter import BBoxPolyCounter
        from pytissueoptics.scene.geometry.tree.splitUtils.axisSelector import RotateAxis
        self._polyCounter = BBoxPolyCounter()
        self._axisSelector = RotateAxis()
        self._nodeSplitter = CentroidNodeSplitter(self._polyCounter)


class SAHKDTreeStrategy(TreeStrategy):
    def loadComponents(self):
        pass
