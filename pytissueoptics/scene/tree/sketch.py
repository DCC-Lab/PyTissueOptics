from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere
from pytissueoptics.scene.tree.binary.binaryTreeStrategy import BasicKDTreeStrategy
from pytissueoptics.scene.tree.binary import BinaryTree
from pytissueoptics.scene.viewer import MayaviViewer
import time

cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(1, 0, 0))
cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
sphere = Sphere(position=Vector(3, 3, 3), order=4)
scene = Scene([cuboid1, cuboid2, sphere])
t0 = time.time()
kdTree = BinaryTree(scene=scene, treeStrategy=BasicKDTreeStrategy(), maxDepth=7)
t1 = time.time()
print(t1-t0)

bboxes = kdTree.getLeafBoundingBoxesAsCuboids()
print(len(bboxes))

viewer = MayaviViewer()
viewer.add(*scene.getSolids(), representation="mesh")
viewer.add(*bboxes, representation="surface", lineWidth=2, opacity=0.25)
viewer.show()
