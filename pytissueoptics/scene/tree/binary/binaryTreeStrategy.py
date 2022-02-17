from typing import List, Tuple
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.binary.splitUtils import PolyCounter, AxisSelector, NodeSplitter


class BinaryTreeStrategy:
    def __init__(self):
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None
        self._polygons = None
        self._nodeAxis = None
        self._nodeBbox = None
        self._splitLine = None
        self._splitAxis = None
        self._stopCondition = 0
        self._goingLeft = []
        self._goingRight = []
        self._loadComponents()

    def _loadComponents(self):
        raise NotImplementedError

    def run(self,
            polygons: List[Polygon],
            nodeAxis: str,
            nodeBbox: BoundingBox)-> Tuple[int, str, float, List[Polygon], List[Polygon]]:

        self._polygons = polygons
        self._nodeAxis = nodeAxis
        self._nodeBbox = nodeBbox
        self._goingLeft = []
        self._goingRight = []
        self._selectSplitAxis()
        self._split()
        return self._stopCondition, self._splitAxis, self._splitLine, self._goingLeft, self._goingRight


    def _split(self):
        self._stopCondition, self._splitLine, self._goingLeft, self._goingRight = self._nodeSplitter.run(self._splitAxis, self._polygons, self._nodeBbox)

    def _selectSplitAxis(self):
        self._splitAxis = self._axisSelector.run(self._nodeAxis, self._polygons)


class BasicKDTreeStrategy(BinaryTreeStrategy):
    def _loadComponents(self):
        from pytissueoptics.scene.tree.binary.splitUtils.nodeSplitter import DumbSAHSplitter
        from pytissueoptics.scene.tree.binary.splitUtils.polyCounter import BBoxPolyCounter
        from pytissueoptics.scene.tree.binary.splitUtils.axisSelector import RotateAxis
        self._polyCounter = BBoxPolyCounter()
        self._axisSelector = RotateAxis()
        self._nodeSplitter = DumbSAHSplitter(self._polyCounter)


class SAHKDTreeStrategy(BinaryTreeStrategy):
    def _loadComponents(self):
        pass
