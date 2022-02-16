import unittest
from pytissueoptics.scene.geometry.tree.kdtree import KDNode
from pytissueoptics.scene.geometry.tree.treeStrategy import MeanCentroidSplitter
from pytissueoptics.scene.geometry import Vector, Triangle
from pytissueoptics.scene.viewer import MayaviViewer


class TestKDTree(unittest.TestCase):

    def setUp(self):
        polygon1 = Triangle(Vector(1, 1, 1), Vector(0, 0, 1), Vector(-1, -2, 1))
        polygon2 = Triangle(Vector(-4, -6, -6), Vector(-3, -3, -3), Vector(-4, -4, -4))
        self.root = KDNode(parent=None, polygons=[polygon1, polygon2], splitStrategy=MeanCentroidSplitter())

    def testShouldNotExceedMaxDepth(self):
        pass


if __name__ == "__main__":

    viewer = MayaviViewer()
    viewer.show()
