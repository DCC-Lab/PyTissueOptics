from pytissueoptics.scene.tree.treeConstructor.binary.fastBinaryTreeConstructor import FastBinaryTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor, \
    ShrankBoxSAHWideAxisTreeConstructor
from pytissueoptics.scene.intersection import FastIntersectionFinder, SimpleIntersectionFinder, Ray
from pytissueoptics.scene.tests.scene.benchmarkScenes import *
from pytissueoptics.scene.viewer import MayaviViewer
import pandas
import numpy as np
import time

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1200)


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
constructors = [FastBinaryTreeConstructor(), ShrankBoxSAHWideAxisTreeConstructor(), SAHWideAxisTreeConstructor()]
rayAmount = [100, 1000]  # simple vs fast amounts
factor = rayAmount[1] / rayAmount[0]
displayViewer = False

# ================ Profiler Data ===================
dfs = pandas.DataFrame(
    columns=["scene", "name", "fast time", "build time", "simple time", "node count", "leaf count", "AVG Depth",
             "AVG Size", "polygonCount"])
constructionTimes = []
spacePartitions = []
fastTraversalTime = []
simpleTraversalTime = []
count = 0

for j, scene in enumerate(scenes):
    sceneBbox = scene.getBoundingBox()
    constructionTimes.append([])
    fastTraversalTime.append([])
    spacePartitions.append([])
    simpleIntersectionFinder = SimpleIntersectionFinder(scene)
    print(f"====== {scene.__class__.__name__:.15s} ======")
    for c, constructor in enumerate(constructors):
        if c == 0:
            count += 1
            origin_xs = np.random.uniform(sceneBbox.xMin, sceneBbox.xMax, rayAmount[0])
            origin_ys = np.random.uniform(sceneBbox.yMin, sceneBbox.yMax, rayAmount[0])
            origin_zs = np.random.uniform(sceneBbox.zMin, sceneBbox.zMax, rayAmount[0])
            direction_xs = np.random.uniform(-1, 1, rayAmount[0])
            direction_ys = np.random.uniform(-1, 1, rayAmount[0])
            direction_zs = np.random.uniform(-1, 1, rayAmount[0])
            t0 = time.time()
            for i in range(rayAmount[0]):
                ray = Ray(origin=Vector(origin_xs[i], origin_ys[i], origin_zs[i]),
                          direction=Vector(direction_xs[i], direction_ys[i], direction_zs[i]))
                simpleIntersectionFinder.findIntersection(ray)
            t1 = time.time()
            simpleTraversalTime.append(t1 - t0)
            print(
                f"SimpleIntersect - {(count * 100) / (len(scenes) * (len(constructors) + 1)):.2f}% - {simpleTraversalTime[j] * factor:.2f}s")

        count += 1
        origin_xs = np.random.uniform(sceneBbox.xMin, sceneBbox.xMax, rayAmount[1])
        origin_ys = np.random.uniform(sceneBbox.yMin, sceneBbox.yMax, rayAmount[1])
        origin_zs = np.random.uniform(sceneBbox.zMin, sceneBbox.zMax, rayAmount[1])
        direction_xs = np.random.uniform(-1, 1, rayAmount[1])
        direction_ys = np.random.uniform(-1, 1, rayAmount[1])
        direction_zs = np.random.uniform(-1, 1, rayAmount[1])

        constructionInitTime = time.time()
        fastIntersectionFinder = FastIntersectionFinder(scene, constructor=constructor, maxDepth=100, minLeafSize=0)
        constructionTimes[j].append(time.time() - constructionInitTime)

        traversalInit = time.time()
        for i in range(rayAmount[1]):
            ray = Ray(origin=Vector(origin_xs[i], origin_ys[i], origin_zs[i]),
                      direction=Vector(direction_xs[i], direction_ys[i], direction_zs[i]))
            fastIntersectionFinder.findIntersection(ray)
        fastTraversalTime[j].append(time.time() - traversalInit)

        partition = fastIntersectionFinder._partition
        spacePartitions[j].append(partition)
        print(
            f"{constructor.__class__.__name__:^12.15s} - {(count * 100) / (len(scenes) * (len(constructors)+1)):.2f}% - {fastTraversalTime[j][c]:.2f}s - Improvement {((100 * simpleTraversalTime[j] * factor) / fastTraversalTime[j][c]) - 100:.2f}%")
        dfs.loc[dfs.shape[0]] = [f"{scene.__class__.__name__}", f"{constructor.__class__.__name__:^12.15s}",
                                 f"{fastTraversalTime[j][c]:^12.2f}", f"{constructionTimes[j][c]:^12.2f}",
                                 f"{simpleTraversalTime[j]:^12.2f}", f"{partition.getNodeCount():^12}",
                                 f"{partition.getLeafCount():^12}", f"{partition.getAverageLeafDepth():^12.2f}",
                                 f"{partition.getAverageLeafSize():^12.2f}", f"{len(scene.getPolygons()):^12}"]

print("\n\n")
dfs[['scene', 'name', 'build time']].to_csv("data2.csv")

if displayViewer:
    viewer = MayaviViewer()
    for j, scene in enumerate(scenes):
        for partition in spacePartitions[j]:
            bBoxes = partition.getLeafBoundingBoxesAsCuboids()
            viewer = MayaviViewer()
            viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05, opacity=0.5)
            viewer.add(*bBoxes, representation="wireframe", lineWidth=3, color=(1, 0, 0), opacity=0.7)
            viewer.show()
            viewer.clear()
