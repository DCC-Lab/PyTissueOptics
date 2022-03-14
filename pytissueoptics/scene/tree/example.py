from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere, Cube
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor, ShrankBoxSAHWideAxisTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor, SAHBasicKDTreeConstructor
from pytissueoptics.scene.intersection import FastIntersectionFinder, SimpleIntersectionFinder, Ray
from pytissueoptics.scene.viewer import MayaviViewer

import numpy as np
import time


class PhantomScene(Scene):
    def __init__(self):
        super().__init__()
        self._solids = self._createSolids()

    def _createSolids(self):
        w, d, h, t = 20, 20, 8, 0.1
        floor = Cuboid(w + t, t, d + t, position=Vector(0, -t / 2, 0))
        leftWall = Cuboid(t, h, d, position=Vector(-w / 2, h / 2, 0))
        rightWall = Cuboid(t, h, d, position=Vector(w / 2, h / 2, 0))
        backWall = Cuboid(w, h, t, position=Vector(0, h / 2, -d / 2))
        cube = Cube(3, position=Vector(-5, 3 / 2, -6))
        return [floor, leftWall, rightWall, backWall, cube]


cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(4, -2, 6))
cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
sphere = Sphere(position=Vector(3, 3, 3), order=3)
#scene = Scene([cuboid1, cuboid2, sphere])
scene = PhantomScene()

constructors = [BalancedKDTreeConstructor(), SAHBasicKDTreeConstructor(), SAHWideAxisTreeConstructor(), ShrankBoxSAHWideAxisTreeConstructor()]
constructionTimes = []
spacePartitions = []
randomIntersections = [100, 1000, 10000]
fastTraversalTime = []
simpleTraversalTime = []

for c, constructor in enumerate(constructors):
    t0 = time.time()
    partition = SpacePartition(scene.getBoundingBox(), scene.getPolygons(), constructor=constructor,
                            maxDepth=100, minLeafSize=6)
    t1 = time.time()
    constructionTimes.append(t1-t0)
    spacePartitions.append(partition)
    fastTraversalTime.append([])
    simpleTraversalTime.append([])
    simpleIntersectionFinder = SimpleIntersectionFinder(scene)
    fastIntersectionFinder = FastIntersectionFinder(scene, constructor=constructor, maxDepth=100, minLeafSize=6)
    sceneBbox = scene.getBoundingBox()
    for intersectionAmount in randomIntersections:
        origin_xs = np.random.uniform(sceneBbox.xMin, sceneBbox.xMax, intersectionAmount)
        origin_ys = np.random.uniform(sceneBbox.yMin, sceneBbox.yMax, intersectionAmount)
        origin_zs = np.random.uniform(sceneBbox.zMin, sceneBbox.zMax, intersectionAmount)
        direction_xs = np.random.uniform(-1, 1, intersectionAmount)
        direction_ys = np.random.uniform(-1, 1, intersectionAmount)
        direction_zs = np.random.uniform(-1, 1, intersectionAmount)

        t0 = time.time()
        for i in range(intersectionAmount):
            ray = Ray(origin=Vector(origin_xs[i], origin_ys[i], origin_zs[i]), direction=Vector(direction_xs[i], direction_ys[i], direction_zs[i]))
            fastIntersectionFinder.findIntersection(ray)
        t1 = time.time()
        fastTraversalTime[c].append(t1-t0)

        t0 = time.time()
        for i in range(intersectionAmount):
            ray = Ray(origin=Vector(origin_xs[i], origin_ys[i], origin_zs[i]),
                      direction=Vector(direction_xs[i], direction_ys[i], direction_zs[i]))
            simpleIntersectionFinder.findIntersection(ray)
        t1 = time.time()
        simpleTraversalTime[c].append(t1 - t0)

    print(f"\nStrategy:{partition._constructor.__class__.__name__}\n"
        f"Scene Poly Count:{len(scene.getPolygons())}\n"
        f"Avg Leaf Size:{partition.getAverageLeafSize()}\n"
        f"Avf Leaf Depth:{partition.getAverageLeafDepth()}\n"
        f"Max Leaf Depth:{partition.getMaxLeafDepth()}\n"
        f"Total Node:{partition.getNodeCount()}\n"
        f"Total Leaf Node:{partition.getLeafCount()}\n"
        f"Tree Render Time:{constructionTimes[c]}s\n"
        f"Fast Tree Traversal Time:{fastTraversalTime[c]}s\n"
        f"Simple Tree Traversal Time:{simpleTraversalTime[c]}s\n")


#
viewer = MayaviViewer()
for partition in spacePartitions:
    bBoxes = partition.getLeafBoundingBoxesAsCuboids()
    viewer = MayaviViewer()
    viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05, opacity=0.5)
    viewer.add(*bBoxes, representation="wireframe", lineWidth=0.2, color=(1, 0, 0))
    viewer.show()
    viewer.clear()

