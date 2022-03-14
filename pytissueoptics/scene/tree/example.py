from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor, ShrankBoxSAHWideAxisTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor, SAHBasicKDTreeConstructor
from pytissueoptics.scene.viewer import MayaviViewer

import time


cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(4, -2, 6))
cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
sphere = Sphere(position=Vector(3, 3, 3), order=3)
scene = Scene([cuboid1, cuboid2, sphere])


constructors = [SAHBasicKDTreeConstructor(), SAHWideAxisTreeConstructor(), ShrankBoxSAHWideAxisTreeConstructor()]
deltaT = []
spacePartitions = []

for constructor in constructors:
    t0 = time.time()
    partition = SpacePartition(scene.getBoundingBox(), scene.getPolygons(), constructor=constructor,
                            maxDepth=100, minLeafSize=6)
    t1 = time.time()

    deltaT.append(t1-t0)
    spacePartitions.append(partition)



for i, partition in enumerate(spacePartitions):
      print(f"\nStrategy:{partition._constructor.__class__.__name__}\n"
            f"Scene Poly Count:{len(scene.getPolygons())}\n"
            f"Avg Leaf Size:{partition.getAverageLeafSize()}\n"
            f"Avf Leaf Depth:{partition.getAverageLeafDepth()}\n"
            f"Max Leaf Depth:{partition.getMaxLeafDepth()}\n"
            f"Total Node:{partition.getNodeCount()}\n"
            f"Total Leaf Node:{partition.getLeafCount()}\n"
            f"Tree Render Time:{deltaT[i]}s\n")


viewer = MayaviViewer()
for partition in spacePartitions:
    bBoxes = partition.getLeafBoundingBoxesAsCuboids()
    viewer = MayaviViewer()
    viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05, opacity=0.5)
    viewer.add(*bBoxes, representation="wireframe", lineWidth=0.2, color=(1, 0, 0))
    viewer.show()
    viewer.clear()
