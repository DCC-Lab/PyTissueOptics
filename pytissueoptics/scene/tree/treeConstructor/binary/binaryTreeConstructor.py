from pytissueoptics.scene.tree import TreeConstructor


class SAHBasicKDTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHBasicKDTreeConstructor, self).__init__()
        from pytissueoptics.scene.tree.treeConstructor.binary.binaryNodeSplitter import HardSAHNodeSplitter
        from pytissueoptics.scene.tree.treeConstructor.binary.binaryPolyCounter import CentroidPolyCounter
        from pytissueoptics.scene.tree.treeConstructor.binary.binaryAxisSelector import RotateAxis
        self._polyCounter = CentroidPolyCounter()
        self._axisSelector = RotateAxis()
        self._nodeSplitter = HardSAHNodeSplitter(self._polyCounter)


class SAHWideAxisTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHWideAxisTreeConstructor, self).__init__()
        from pytissueoptics.scene.tree.treeConstructor.binary.binaryNodeSplitter import HardSAHNodeSplitter
        from pytissueoptics.scene.tree.treeConstructor.binary.binaryPolyCounter import BBoxPolyCounter
        from pytissueoptics.scene.tree.treeConstructor.binary.binaryAxisSelector import LargestSpanAxis
        self._polyCounter = BBoxPolyCounter()
        self._axisSelector = LargestSpanAxis()
        self._nodeSplitter = HardSAHNodeSplitter(self._polyCounter, nbOfSplitPlane=50, splitCostPercentage=0.1)
