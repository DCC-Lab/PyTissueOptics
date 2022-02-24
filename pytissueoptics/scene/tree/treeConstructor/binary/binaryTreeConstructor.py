from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.binaryNodeSplitter import HardSAHNodeSplitter
from pytissueoptics.scene.tree.treeConstructor.binary.binaryPolyCounter import BBoxPolyCounter, CentroidPolyCounter
from pytissueoptics.scene.tree.treeConstructor.binary.binaryAxisSelector import LargestSpanAxis, RotatingAxis


class SAHBasicKDTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHBasicKDTreeConstructor, self).__init__()
        self.setContext(RotatingAxis(), CentroidPolyCounter(),
                        HardSAHNodeSplitter(nbOfSplitPlanes=50, splitCostPercentage=0.1))


class SAHWideAxisTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHWideAxisTreeConstructor, self).__init__()
        self.setContext(LargestSpanAxis(), BBoxPolyCounter(),
                        HardSAHNodeSplitter(nbOfSplitPlanes=50, splitCostPercentage=0.1))
