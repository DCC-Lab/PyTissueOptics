from pytissueoptics.scene.tree import TreeStrategy


class BinaryTreeStrategy(TreeStrategy):
    def _loadComponents(self):
        raise NotImplementedError


class BasicKDTreeStrategy(BinaryTreeStrategy):
    def _loadComponents(self):
        from pytissueoptics.scene.tree.binary.binaryNodeSplitter import CentroidNodeSplitter, HardSAHSplitter
        from pytissueoptics.scene.tree.binary.BinaryPolyCounter import BBoxPolyCounter, CentroidPolyCounter
        from pytissueoptics.scene.tree.binary.binaryAxisSelector import RotateAxis, LargestSpanAxis
        self._polyCounter = BBoxPolyCounter()
        self._axisSelector = RotateAxis()
        self._nodeSplitter = HardSAHSplitter(self._polyCounter)
