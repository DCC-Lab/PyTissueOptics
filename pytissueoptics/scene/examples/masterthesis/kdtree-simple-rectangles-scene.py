from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitThreeAxesConstructor
from pytissueoptics.scene.viewer import MayaviViewer
import numpy as np
import time


cuboid1 = Cuboid(1, 1, 1, position=Vector(5, 5, 5))
cuboid2 = Cuboid(1, 1, 2, position=Vector(0, 5, 0))
cuboid3 = Cuboid(1, 1, 2, position=Vector(0, 0, 0))
cuboid4 = Cuboid(1, 1, 2, position=Vector(0, 0, 5))
cuboid5 = Sphere(position=Vector(2, 2, 2))
cuboid5.rotate(45, 45, 45)
scene = Scene([cuboid1, cuboid2, cuboid3, cuboid4, cuboid5])

t0 = time.time()
kdTree = SpacePartition(scene.getBoundingBox(), scene.getPolygons(), constructor=NoSplitThreeAxesConstructor(),
                        maxDepth=10, minLeafSize=2)
t1 = time.time()

bBoxes = kdTree.getAllSubNodeBoundingBoxesAsCuboids()

print(f"Scene Poly Count:{len(scene.getPolygons())}\n"
      f"Min Leaf Size:{kdTree._minLeafSize}\n"
      f"Max Tree Depth:{kdTree._maxDepth}\n"
      f"Total Node:{kdTree.getNodeCount()}\n"
      f"Total Leaf Node:{kdTree.getLeafCount()}\n"
      f"Tree Render Time:{t1 - t0}s")

viewer = MayaviViewer()
mayaviSolids = viewer.add(*scene.getSolids(), representation="wireframe", lineWidth=0.1)
mayaviBoxes = viewer.add(*bBoxes, representation="surface", lineWidth=25, opacity=0.1)
sceneProp = mayaviSolids[0].parent.parent.parent.parent
sceneProp.scene.background = (1, 1, 1)

for s in mayaviSolids:
      s.actor.property.representation = "wireframe"
      s.module_manager.scalar_lut_manager.lut.table = np.array([[25, 152, 158, 255]]*255)
      # s.actor.property.line_as_tube
      s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
      s.actor.property.edge_visibility = True
      s.actor.property.line_width = 5
      s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
      s.actor.property.interpolation = "flat"


for y in mayaviBoxes:
      y.actor.property.representation = "surface"
      y.actor.property.opacity = 0.1
      y.module_manager.scalar_lut_manager.lut.table = np.array([[255, 15, 15, 255]] * 255)
#       y.actor.property.line_as_tube
      y.actor.property.edge_visibility = False
      y.actor.property.line_width = 25
      y.actor.property.edge_color = (0, 0, 0)
      y.actor.property.interpolation = "flat"

viewer.show()
