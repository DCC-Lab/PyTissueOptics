from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor, ShrankBoxSAHWideAxisTreeConstructor
from pytissueoptics.scene.viewer import MayaviViewer

import time


cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(4, -2, 6))
cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
sphere = Sphere(position=Vector(3, 3, 3), order=3)
scene = Scene([cuboid1, cuboid2, sphere])

t0 = time.time()

kdTree = SpacePartition(scene.getBoundingBox(), scene.getPolygons(), constructor=ShrankBoxSAHWideAxisTreeConstructor(),
                        maxDepth=16, minLeafSize=6)
t1 = time.time()
kdTree2 = SpacePartition(scene.getBoundingBox(), scene.getPolygons(), constructor=SAHWideAxisTreeConstructor(),
                        maxDepth=16, minLeafSize=6)
t2 = time.time()

t = [t0, t1, t2]
trees = [kdTree, kdTree2]

for i, tree in enumerate(trees):
      print(f"Scene Poly Count:{len(scene.getPolygons())}\n"
            f"Min Leaf Size:{tree.minLeafSize}\n"
            f"Max Tree Depth:{tree.maxDepth}\n"
            f"Total Node:{tree.getNodeCount()}\n"
            f"Total Leaf Node:{tree.getLeafCount()}\n"
            f"Tree Render Time:{t[i+1] - t[i]}s")



bBoxes = kdTree.getLeafBoundingBoxesAsCuboids()
bBoxes2 = kdTree2.getLeafBoundingBoxesAsCuboids()
viewer = MayaviViewer()
#viewer.add(*scene.getSolids(), representation="mesh", lineWidth=0.1)
viewer.add(*bBoxes, representation="surface", lineWidth=0.1, opacity=0.25, color=(1, 0, 0))
#viewer.add(*bBoxes2, representation="surface", lineWidth=0.1, opacity=0.25, color=(0, 1, 0))
viewer.show()
