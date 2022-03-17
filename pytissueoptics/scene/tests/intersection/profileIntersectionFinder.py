from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid, Sphere, Cube, Ellipsoid
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor, ShrankBoxSAHWideAxisTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor, SAHBasicKDTreeConstructor
from pytissueoptics.scene.intersection import FastIntersectionFinder, SimpleIntersectionFinder, Ray
from pytissueoptics.scene.viewer import MayaviViewer

import numpy as np
import time


class PhantomScene(Scene):
    ROOM = []
    CROSSWALK = []
    OBJECTS = []
    SIGN = []

    def __init__(self):
        super().__init__()
        self._create()
        self._solids = [*self.ROOM, *self.CROSSWALK, *self.OBJECTS, *self.SIGN]

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.ROOM[:-1], representation="surface", colormap="bone")
        viewer.add(self.ROOM[-1], representation="surface", colormap="bone", reverseColormap=True)
        viewer.add(*self.CROSSWALK, *self.OBJECTS, *self.SIGN, representation="surface", colormap="Set2",
                   reverseColormap=True, constantColor=True)

    def _create(self):
        self.ROOM = self._room()
        self.CROSSWALK = self._crossWalk()
        self.OBJECTS = self._objects()
        self.SIGN = self._sign()

    def _room(self):
        w, d, h, t = 20, 20, 8, 0.1
        floor = Cuboid(w + t, t, d + t, position=Vector(0, -t / 2, 0))
        leftWall = Cuboid(t, h, d, position=Vector(-w / 2, h / 2, 0))
        rightWall = Cuboid(t, h, d, position=Vector(w / 2, h / 2, 0))
        backWall = Cuboid(w, h, t, position=Vector(0, h / 2, -d / 2))
        return [floor, leftWall, rightWall, backWall]

    def _crossWalk(self):
        return [Cuboid(0.7, 0.001, 4, position=Vector(i, 0, -8)) for i in range(-5, 5)]

    def _objects(self):
        cubeA = Cube(3, position=Vector(-5, 3/2, -6))
        cubeB = Cube(3, position=Vector(5, 3/2, -6))
        cubeB.rotate(0, 20, 0)
        cubeC = Cube(1, position=Vector(-5, 3.866, -6))
        cubeC.rotate(0, 0, 45)
        cubeC.rotate(45, 0, 0)
        sphere = Sphere(0.75, order=2, position=Vector(5, 3.75, -6))
        return [cubeA, cubeB, cubeC, sphere]

    def _sign(self):
        sign = Cuboid(1.5, 1.5, 0.001, position=Vector(7.8, 5, -5 + (0.1 + 0.01)/2))
        sign.rotate(0, 0, 45)
        stand = Cuboid(0.1, 5, 0.1, position=Vector(7.8, 2.5, -5))
        return [sign, stand]

class RandomShapesScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(4, -2, 6))
        cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
        sphere1 = Sphere(position=Vector(3, 3, 3), order=3)
        ellipsoid1 = Ellipsoid(1, 2, 3, position=Vector(10, 3, -3), order=3)
        self._solids.extend([cuboid1, cuboid2, sphere1, ellipsoid1])

class XAlignedSpheres(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere1 = Sphere(position=Vector(2, 0, 0), order=3)
        sphere2 = Sphere(position=Vector(5, 0, 0), order=3)
        sphere3 = Sphere(position=Vector(9, 0, 0), order=3)
        sphere4 = Sphere(position=Vector(20, 0, 0), order=3)
        self._solids.extend([sphere1, sphere2, sphere3, sphere4])

class ZAlignedSpheres(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere1 = Sphere(position=Vector(0, 0, 2), order=2)
        sphere2 = Sphere(position=Vector(0, 0, 5), order=2)
        sphere3 = Sphere(position=Vector(0, 0, 9), order=2)
        sphere4 = Sphere(position=Vector(0, 0, 20), order=2)
        self._solids.extend([sphere1, sphere2, sphere3, sphere4])

class DiagonallyAlignedSpheres(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere1 = Sphere(position=Vector(0, 0, 0), order=2)
        sphere2 = Sphere(position=Vector(3, 3, 3), order=2)
        sphere3 = Sphere(position=Vector(7, 7, 7), order=2)
        sphere4 = Sphere(position=Vector(20, 20, 20), order=2)
        self._solids.extend([sphere1, sphere2, sphere3, sphere4])


scene1 = PhantomScene()
scene2 = RandomShapesScene()
scene3 = XAlignedSpheres()
scene4 = ZAlignedSpheres()
scene5 = DiagonallyAlignedSpheres()

# scenes = [scene1, scene2, scene3, scene4, scene5]
scenes = [scene3]
for j, scene in enumerate(scenes):
    print(f"\n=============SCENE #{j}:{scene.__class__.__name__}============\n")
    constructors = [BalancedKDTreeConstructor(), SAHBasicKDTreeConstructor(), SAHWideAxisTreeConstructor(), ShrankBoxSAHWideAxisTreeConstructor()]
    constructionTimes = []
    spacePartitions = []
    randomIntersections = [10000]
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
            # origin_xs = np.random.uniform(sceneBbox.xMin, sceneBbox.xMax, intersectionAmount)
            # origin_ys = np.random.uniform(sceneBbox.yMin, sceneBbox.yMax, intersectionAmount)
            # origin_zs = np.random.uniform(sceneBbox.zMin, sceneBbox.zMax, intersectionAmount)
            # direction_xs = np.random.uniform(-1, 1, intersectionAmount)
            # direction_ys = np.random.uniform(-1, 1, intersectionAmount)
            # direction_zs = np.random.uniform(-1, 1, intersectionAmount)

            t0 = time.time()
            for i in range(intersectionAmount):
                ray = Ray(origin=Vector(0, 0, 0), direction=Vector(1, 0, 0))
                fastIntersectionFinder.findIntersection(ray)
            t1 = time.time()
            fastTraversalTime[c].append(t1-t0)

            t0 = time.time()
            for i in range(intersectionAmount):
                ray = Ray(origin=Vector(0, 0, 0),
                          direction=Vector(1, 0, 0))
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
# viewer = MayaviViewer()
# for partition in spacePartitions:
#     bBoxes = partition.getLeafBoundingBoxesAsCuboids()
#     viewer = MayaviViewer()
#     viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05, opacity=0.5)
#     viewer.add(*bBoxes, representation="wireframe", lineWidth=0.2, color=(1, 0, 0))
#     viewer.show()
#     viewer.clear()

