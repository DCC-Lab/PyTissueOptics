from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.binaryNodeSplitter import HardSAHNodeSplitter,\
    MeanCentroidNodeSplitter, ShrankBoxSAHNodeSplitter
from pytissueoptics.scene.tree.treeConstructor.binary.binaryPolyCounter import BBoxPolyCounter, CentroidPolyCounter
from pytissueoptics.scene.tree.treeConstructor.binary.binaryAxisSelector import LargestSpanAxis, RotatingAxis



class BalancedKDTreeConstructor(TreeConstructor):
    def __init__(self):
        super(BalancedKDTreeConstructor, self).__init__()
        self.setContext(RotatingAxis(), CentroidPolyCounter(), MeanCentroidNodeSplitter())


class SAHBasicKDTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHBasicKDTreeConstructor, self).__init__()
        self.setContext(RotatingAxis(), CentroidPolyCounter(),
                        HardSAHNodeSplitter(nbOfSplitPlanes=20, splitCostPercentage=0.1))


class SAHWideAxisTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHWideAxisTreeConstructor, self).__init__()
        self.setContext(LargestSpanAxis(), BBoxPolyCounter(),
                        HardSAHNodeSplitter(nbOfSplitPlanes=20, splitCostPercentage=0.1))


class SAHWideAxisCentroidTreeConstructor(TreeConstructor):
    def __init__(self):
        super(SAHWideAxisCentroidTreeConstructor, self).__init__()
        self.setContext(LargestSpanAxis(), CentroidPolyCounter(),
                        HardSAHNodeSplitter(nbOfSplitPlanes=20, splitCostPercentage=0.1))


class ShrankBoxSAHWideAxisTreeConstructor(TreeConstructor):
    def __init__(self):
        super(ShrankBoxSAHWideAxisTreeConstructor, self).__init__()
        self.setContext(LargestSpanAxis(), BBoxPolyCounter(),
                        ShrankBoxSAHNodeSplitter(nbOfSplitPlanes=20, splitCostPercentage=0.1))


class ShrankBoxSAHWideAxisCentroidTreeConstructor(TreeConstructor):
    def __init__(self):
        super(ShrankBoxSAHWideAxisCentroidTreeConstructor, self).__init__()
        self.setContext(LargestSpanAxis(), CentroidPolyCounter(),
                        ShrankBoxSAHNodeSplitter(nbOfSplitPlanes=20, splitCostPercentage=0.1))