from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector, Polygon, SurfaceCollection, BoundingBox
from pytissueoptics.scene.solids import Cuboid, Sphere, Cube, Ellipsoid, Solid
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.fastBinaryTreeConstructor import FastBinaryTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor, ShrankBoxSAHWideAxisTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import BalancedKDTreeConstructor, SAHBasicKDTreeConstructor
from pytissueoptics.scene.intersection import FastIntersectionFinder, SimpleIntersectionFinder, Ray
from pytissueoptics.scene.viewer import MayaviViewer
import pandas
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1200)
import numpy as np
import time


class AAxisAlignedPolygonScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        vertices = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0)]
        polygon = Polygon(vertices=vertices)
        aSolid = Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=SurfaceCollection({"lonely": [polygon]}))
        aSolid._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([aSolid])

class APolygonScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        vertices = [Vector(0, 0, 0), Vector(1, 0, 1), Vector(-0.2, -0.5, 3)]
        polygon = Polygon(vertices=vertices)
        aSolid = Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=SurfaceCollection({"lonely": [polygon]}))
        aSolid._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([aSolid])

class ACubeScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(4, -2, 6))
        cuboid1._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([cuboid1])

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
        sphere1 = Sphere(position=Vector(3, 3, 3), order=1)
        ellipsoid1 = Ellipsoid(1, 2, 3, position=Vector(10, 3, -3), order=2)
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


scene000 = AAxisAlignedPolygonScene()
scene00 = APolygonScene()
scene0 = ACubeScene()
scene1 = PhantomScene()
scene2 = RandomShapesScene()
scene3 = XAlignedSpheres()
scene4 = ZAlignedSpheres()
scene5 = DiagonallyAlignedSpheres()

# ========= Profiler Important Parameters ================
# scenes = [scene1, scene2, scene3, scene4, scene5]
# scenes = [scene000, scene00, scene0]
# scenes = [scene2]
scenes = [scene000, scene00, scene0, scene1, scene2, scene3, scene4, scene5]
# constructors = [BalancedKDTreeConstructor(), SAHBasicKDTreeConstructor(),
# SAHWideAxisTreeConstructor(), ShrankBoxSAHWideAxisTreeConstructor()]
constructors = [FastBinaryTreeConstructor(), ShrankBoxSAHWideAxisTreeConstructor(), SAHWideAxisTreeConstructor()]
randomIntersections = [1000, 100000]  # simple vs fast amounts


# ================ Profiler Data ===================
dfs = pandas.DataFrame(columns=["scene", "name", "fast time", "build time", "simple time", "node count", "leaf count", "AVG Depth", "AVG Size", "polygonCount"])
constructionTimes = []
spacePartitions = []
fastTraversalTime = []
simpleTraversalTime = []
count = 0

for j, scene in enumerate(scenes):
    sceneBbox = scene.getBoundingBox()
    dfs.loc[dfs.shape[0]] = (f"{scene.__class__.__name__}", "", "", "", "", "", "", "", "", "")
    constructionTimes.append([])
    fastTraversalTime.append([])
    simpleIntersectionFinder = SimpleIntersectionFinder(scene)
    print(f"====== {scene.__class__.__name__:.15s} ======")
    for c, constructor in enumerate(constructors):
        if c == 0:
            count += 1
            origin_xs = np.random.uniform(sceneBbox.xMin, sceneBbox.xMax, randomIntersections[0])
            origin_ys = np.random.uniform(sceneBbox.yMin, sceneBbox.yMax, randomIntersections[0])
            origin_zs = np.random.uniform(sceneBbox.zMin, sceneBbox.zMax, randomIntersections[0])
            direction_xs = np.random.uniform(-1, 1, randomIntersections[0])
            direction_ys = np.random.uniform(-1, 1, randomIntersections[0])
            direction_zs = np.random.uniform(-1, 1, randomIntersections[0])
            t0 = time.time()
            for i in range(randomIntersections[0]):
                ray = Ray(origin=Vector(origin_xs[i], origin_ys[i], origin_zs[i]),
                          direction=Vector(direction_xs[i], direction_ys[i], direction_zs[i]))
                simpleIntersectionFinder.findIntersection(ray)
            t1 = time.time()
            simpleTraversalTime.append(t1 - t0)
            print(
                f"SimpleIntersect - {(count * 100) / (len(scenes) * (len(constructors)+1)):.2f}% - {simpleTraversalTime[j]:.2f}s - Multiplication Factor {100}")

        count += 1
        origin_xs = np.random.uniform(sceneBbox.xMin, sceneBbox.xMax, randomIntersections[1])
        origin_ys = np.random.uniform(sceneBbox.yMin, sceneBbox.yMax, randomIntersections[1])
        origin_zs = np.random.uniform(sceneBbox.zMin, sceneBbox.zMax, randomIntersections[1])
        direction_xs = np.random.uniform(-1, 1, randomIntersections[1])
        direction_ys = np.random.uniform(-1, 1, randomIntersections[1])
        direction_zs = np.random.uniform(-1, 1, randomIntersections[1])

        constructionInitTime = time.time()
        fastIntersectionFinder = FastIntersectionFinder(scene, constructor=constructor, maxDepth=100, minLeafSize=0)
        constructionTimes[j].append(time.time() - constructionInitTime)

        traversalInit = time.time()
        for i in range(randomIntersections[1]):
            ray = Ray(origin=Vector(origin_xs[i], origin_ys[i], origin_zs[i]), direction=Vector(direction_xs[i], direction_ys[i], direction_zs[i]))
            fastIntersectionFinder.findIntersection(ray)
        fastTraversalTime[j].append(time.time() - traversalInit)

        partition = fastIntersectionFinder._partition
        print(f"{constructor.__class__.__name__:^12.15s} - {(count * 100) / (len(scenes) * len(constructors)):.2f}% - {fastTraversalTime[j][c]:.2f}s - Improvement Factor {1/(fastTraversalTime[j][c]/simpleTraversalTime[j])}")
        dfs.loc[dfs.shape[0]] = ["", f"{constructor.__class__.__name__:^12.15s}",
                                 f"{fastTraversalTime[j][c]:^12.2f}", f"{constructionTimes[j][c]:^12.2f}",
                                 f"{simpleTraversalTime[j]:^12.2f}", f"{partition.getNodeCount():^12}",
                                 f"{partition.getLeafCount():^12}", f"{partition.getAverageLeafDepth():^12.2f}",
                                 f"{partition.getAverageLeafSize():^12.2f}", f"{len(scene.getPolygons()):^12}"]


print("\n\n")
print(dfs)

viewer = MayaviViewer()
for j, scene in enumerate(scenes):
    for partition in spacePartitions[j]:
        bBoxes = partition.getLeafBoundingBoxesAsCuboids()
        viewer = MayaviViewer()
        viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05, opacity=0.5)
        viewer.add(*bBoxes, representation="wireframe", lineWidth=3, color=(1, 0, 0), opacity=0.7)
        viewer.show()
        viewer.clear()
