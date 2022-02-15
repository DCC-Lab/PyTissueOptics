import unittest
from pytissueoptics.scene.geometry.tree.kdtree import KDTree
from pytissueoptics.scene.geometry.tree.kdtree.splitter import MeanCentroidSplitter
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere
from pytissueoptics.scene.viewer import MayaviViewer


class TestKDTree(unittest.TestCase):

    def setUp(self):
        cuboid1 = Cuboid(1, 1, 1, position=Vector(2, 0, 0))
        cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
        scene = Scene([cuboid1, cuboid2])
        self.kdTree = KDTree(scene=scene, splitStrategy=MeanCentroidSplitter(), maxDepth=10)

    def testShouldNotExceedMaxDepth(self):
        pass


if __name__ == "__main__":
    cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(1, 0, 0))
    cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
    sphere = Sphere(position=Vector(3, 3, 3))
    scene = Scene([cuboid1, cuboid2, sphere])
    kdTree = KDTree(scene=scene, splitStrategy=MeanCentroidSplitter(), maxDepth=6)

    bboxes = kdTree.getLeafBoundingBoxesAsCuboids()
    print(len(bboxes))

    viewer = MayaviViewer()
    viewer.add(*scene.getSolids(), representation="mesh")
    viewer.add(*bboxes, representation="surface", lineWidth=2, opacity=0.25)
    viewer.show()
