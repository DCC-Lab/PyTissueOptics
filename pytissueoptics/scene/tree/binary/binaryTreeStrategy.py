from typing import List, Tuple
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree import TreeStrategy


class BinaryTreeStrategy(TreeStrategy):
    def _loadComponents(self):
        raise NotImplementedError


class BasicKDTreeStrategy(BinaryTreeStrategy):
    def _loadComponents(self):
        from pytissueoptics.scene.tree.binary.splitUtils.binaryNodeSplitter import DumbSAHSplitter
        from pytissueoptics.scene.tree.binary.BinaryPolyCounter import BBoxPolyCounter
        from pytissueoptics.scene.tree.binary.splitUtils.binaryAxisSelector import RotateAxis
        self._polyCounter = BBoxPolyCounter()
        self._axisSelector = RotateAxis()
        self._nodeSplitter = DumbSAHSplitter(self._polyCounter)


class SAHKDTreeStrategy(BinaryTreeStrategy):
    def _loadComponents(self):
        pass
