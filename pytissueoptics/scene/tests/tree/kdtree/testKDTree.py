import unittest
from pytissueoptics.scene.geometry.tree.kdtree import KDTree
from pytissueoptics.scene.geometry.tree.treeStrategy import MeanCentroidSplitter
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere
import time


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
    sphere = Sphere(position=Vector(3, 3, 3), order=4)
    scene = Scene([cuboid1, cuboid2, sphere])
    t0 = time.time()
    kdTree = KDTree(scene=scene, splitStrategy=MeanCentroidSplitter(), maxDepth=10)
    t1 = time.time()
    print(t1-t0)

    #bboxes = kdTree.getLeafBoundingBoxesAsCuboids()
    #print(len(bboxes))

    #viewer = MayaviViewer()
    #viewer.add(*scene.getSolids(), representation="mesh")
    #viewer.add(*bboxes, representation="surface", lineWidth=2, opacity=0.25)
    #viewer.show()
