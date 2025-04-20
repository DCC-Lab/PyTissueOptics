import time
from typing import List

import numpy as np
import pandas

from pytissueoptics.scene.geometry import BoundingBox, Vector
from pytissueoptics.scene.intersection import (
    FastIntersectionFinder,
    Ray,
    RaySource,
    SimpleIntersectionFinder,
    UniformRaySource,
)
from pytissueoptics.scene.intersection.intersectionFinder import Intersection
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.solids import Cuboid
from pytissueoptics.scene.tests.scene.benchmarkScenes import (
    AAxisAlignedPolygonScene,
    ACubeScene,
    APolygonScene,
    ASphereScene,
    DiagonallyAlignedSpheres,
    PhantomScene,
    RandomShapesScene,
    TwoCubesScene,
    TwoSpheresScene,
    XAlignedSpheres,
    ZAlignedSpheres,
)
from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.noSplitOneAxisConstructor import NoSplitOneAxisConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.noSplitThreeAxesConstructor import NoSplitThreeAxesConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.splitTreeAxesConstructor import SplitThreeAxesConstructor
from pytissueoptics.scene.viewer import MayaviViewer

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1200)


class IntersectionFinderBenchmark:
    def __init__(self, rayAmount=10000, maxDepth=100, minLeafSize=0, factor=10, constructors=None, displayViewer=False):
        self.scenes = self._getScenes()
        if constructors is None:
            self.constructors = [NoSplitOneAxisConstructor(), NoSplitThreeAxesConstructor(),
                                 SplitThreeAxesConstructor()]
        else:
            self.constructors = constructors
        self.rayAmount = rayAmount
        self.factor = factor
        self.maxDepth = maxDepth
        self.minLeafSize = minLeafSize
        self.displayViewer = displayViewer
        self.partitions = []
        self.simpleTraversalTime = []
        self.count = 0
        self.stats = pandas.DataFrame(
            columns=["scene", "polygon count", "finder polycount", "algo name", "build time", "fast time", "total time",
                     "node count", "leaf count", "avg leaf depth", "avg leaf size", "improvement"])

    @staticmethod
    def _getScenes():
        scene00 = AAxisAlignedPolygonScene()
        scene01 = APolygonScene()
        scene02 = ACubeScene()
        scene03 = TwoCubesScene()
        scene04 = ASphereScene()
        scene05 = TwoSpheresScene()
        scene06 = RandomShapesScene()
        scene1 = PhantomScene()
        scene2 = XAlignedSpheres()
        scene3 = ZAlignedSpheres()
        scene4 = DiagonallyAlignedSpheres()
        return [scene00, scene01, scene02, scene03, scene04, scene05, scene06, scene1, scene2, scene3, scene4]

    def runValidation(self, resolution: int = 100, constructors: List[TreeConstructor] = None,
                      displayFailed: bool = True):
        print(f"{('=' * 125):^125}")
        print(
            f"{str('name'):^20}"
            f" - {str('build'):^10}"
            f" - {str('traversal'):^10}"
            f" - {str('polygons'):^10}"
            f" - {str('nodes'):^10}"
            f" - {str('leaves'):^10}"
            f" - {str('depth'):^10}"
            f" - {str('missed'):^10}"
            f" - {str('validated'):^10}")
        print(f"{('=' * 125):^125}")
        if constructors is not None:
            self.constructors = constructors

        source = UniformRaySource(position=Vector(0, 4, 0), direction=Vector(0, 0, -1), xTheta=360, yTheta=90,
                                  xResolution=int(5.12 * resolution), yResolution=int(2.56 * resolution))
        referenceMissedRays = self._runValidationReference(source, display=displayFailed)
        for constructor in self.constructors:
            self._runValidationForConstructor(constructor, referenceMissedRays, source, display=displayFailed)

    def _runValidationReference(self, source: RaySource, display: bool = True, ):
        logger = Logger()
        scene = self.scenes[7]
        intersectionFinder = SimpleIntersectionFinder(scene)
        missedRays = 0
        t1 = time.time()
        for ray in source.rays:
            intersection = intersectionFinder.findIntersection(ray)
            if not intersection:
                missedRays += 1
                continue
            signal = self._measureSignal(intersection)
            if logger:
                logger.logDataPoint(signal, intersection.position)
        t2 = time.time()
        traversalTime = t2 - t1
        print(
            f"{intersectionFinder.__class__.__name__:^20.20s}"
            f" - {0.00:^10.2f}"
            f" - {traversalTime:^10.2f}"
            f" - {str(' '):^10}"
            f" - {str(' '):^10}"
            f" - {str(' '):^10}"
            f" - {str(' '):^10}"
            f" - {missedRays:^10}"
            f" - {str('REFERENCE'):^10}")
        if display:
            viewer = MayaviViewer()
            viewer.addLogger(logger)
            viewer.show()
        return missedRays

    def _runValidationForConstructor(self, constructor: TreeConstructor, referenceMissed: int, source: RaySource,
                                     display: bool = True):
        logger = Logger()
        scene = self.scenes[7]
        t0 = time.time()
        intersectionFinder = FastIntersectionFinder(scene, constructor=constructor, maxDepth=self.maxDepth,
                                                    minLeafSize=self.minLeafSize)
        t1 = time.time()
        constructionTime = t1 - t0

        missedRays = 0
        for ray in source.rays:
            intersection = intersectionFinder.findIntersection(ray)
            if not intersection:
                missedRays += 1
                continue
            signal = self._measureSignal(intersection)
            if logger:
                logger.logDataPoint(signal, intersection.position)
        t2 = time.time()
        traversalTime = t2 - t1
        partition = intersectionFinder._partition
        print(
            f"{constructor.__class__.__name__:^20.20s}"
            f" - {constructionTime:^10.2f}"
            f" - {traversalTime:^10.2f}"
            f" - {len(partition.getLeafPolygons()):^10}"
            f" - {partition.getNodeCount():^10}"
            f" - {partition.getLeafCount():^10}",
            f" - {partition.getAverageDepth():^10.2f}"
            f" - {missedRays:^10}"
            f" - {missedRays == referenceMissed:^10}")
        if display and missedRays != referenceMissed:
            viewer = MayaviViewer()
            viewer.addLogger(logger)
            viewer.show()

    def runBenchmark(self):
        print("=================== BENCHMARK ====================")
        for scene in self.scenes:
            self.runBenchmarkForScene(scene)

    def runBenchmarkForScene(self, scene: Scene):
        print(f"\n================ {scene.__class__.__name__:.15s} =================")
        print(f"Scene polygonCount: {len(scene.getPolygons())}, Scene Object Count: {len(scene.solids)}")
        print(f"================ {scene.__class__.__name__:.15s} =================")
        self.partitions.append([])
        self.runReferenceBenchmarkForScene(scene)
        for constructor in self.constructors:
            self.runBenchmarkForSceneWithConstructor(scene, constructor)

    def runReferenceBenchmarkForScene(self, scene: Scene):
        self.count += 1
        source = RandomPositionAndOrientationRaySource(int(self.rayAmount / self.factor),
                                                       scene.getBoundingBox().xyzLimits, position=Vector(0, 0, 0))
        intersectionFinder = SimpleIntersectionFinder(scene)
        startTime = time.time()
        for ray in source.rays:
            intersectionFinder.findIntersection(ray)
        endTime = time.time()
        traversalTime = endTime - startTime
        self.simpleTraversalTime.append(traversalTime * self.factor)
        print(
            f"{(self.count * 100) / (len(self.scenes) * (len(self.constructors) + 1)):.2f}% - {intersectionFinder.__class__.__name__:^12.15s}"
            f" - {traversalTime * self.factor:.2f}s"
            f" - Improvement 1.00x")
        self._saveSimpleStats(scene, intersectionFinder, traversalTime * self.factor)

    def runBenchmarkForSceneWithConstructor(self, scene: Scene, constructor: TreeConstructor):
        source = RandomPositionAndOrientationRaySource(self.rayAmount, scene.getBoundingBox().xyzLimits,
                                                       position=Vector(0,0,0))
        self.runBenchmarkForSceneWithConstructorAndSource(scene, constructor, source)

    def runBenchmarkForSceneWithConstructorAndSource(self, scene: Scene, constructor: TreeConstructor,
                                                     source: RaySource):
        self.count += 1
        startTime = time.time()
        intersectionFinder = FastIntersectionFinder(scene, constructor, self.maxDepth, self.minLeafSize)
        endTime = time.time()
        buildTime = endTime - startTime
        startTime = time.time()
        for ray in source.rays:
            intersectionFinder.findIntersection(ray)
        endTime = time.time()
        traversalTime = endTime - startTime
        print(
            f"{(self.count * 100) / (len(self.scenes) * (len(self.constructors) + 1)):.2f}% - {constructor.__class__.__name__:^12.15s}"
            f" - {traversalTime:.2f}s"
            f" - Improvement {((self.simpleTraversalTime[-1]) / traversalTime):.2f}x")
        self._saveFastStats(scene, intersectionFinder, traversalTime, buildTime)

    def _saveSimpleStats(self, scene: Scene, intersectionFinder: SimpleIntersectionFinder, traversalTime: float):
        self.stats.loc[self.stats.shape[0]] = [f"{scene.__class__.__name__}", f"{len(scene.getPolygons()):^12}",
                                               "-",
                                               f"{intersectionFinder.__class__.__name__:^12.15s}",
                                               "-",
                                               f"{traversalTime:^12.2f}",
                                               f"{traversalTime:^12.2f}",
                                               "-",
                                               "-",
                                               "-",
                                               "-",
                                               "-"]

    def _saveFastStats(self, scene: Scene, intersectionFinder: FastIntersectionFinder, traversalTime: float,
                       buildTime: float):
        partition = intersectionFinder._partition
        self.partitions[-1].append(partition)
        self.stats.loc[self.stats.shape[0]] = [f"{scene.__class__.__name__}", f"{len(scene.getPolygons()):^12}",
                                               f"{len(partition.getLeafPolygons()):^12}",
                                               f"{partition._constructor.__class__.__name__:^12.15s}",
                                               f"{buildTime:^12.2f}",
                                               f"{traversalTime:^12.2f}",
                                               f"{buildTime + traversalTime:^12.2f}",
                                               f"{partition.getNodeCount():^12}",
                                               f"{partition.getLeafCount():^12}",
                                               f"{partition.getAverageDepth():^12.2f}",
                                               f"{partition.getAverageLeafSize():^12.2f}",
                                               f"{((self.simpleTraversalTime[-1]) / traversalTime):.1f}"]

    def displayStats(self):
        print(self.stats)

    def displayBenchmarkTreeResults(self, objectsDisplay: bool = True, scenes: List[Scene] = None,
                                    objectsOpacity: float = 0.5):
        viewer = MayaviViewer()
        if scenes is None:
            scenes = self.scenes
        for j, scene in enumerate(scenes):
            for partition in self.partitions[j]:
                bBoxes = self._getCuboidsFromBBoxes(partition.getLeafBoundingBoxes())
                if objectsDisplay:
                    viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05,
                               opacity=objectsOpacity)
                viewer.add(*bBoxes, representation="wireframe", lineWidth=3, color=(1, 0, 0), opacity=0.7)
                viewer.show()
                viewer.clear()

    @staticmethod
    def _measureSignal(intersection: Intersection) -> float:
        if intersection.polygon.insideEnvironment.material is None:
            return 0.125
        return 1.0

    @staticmethod
    def _getCuboidsFromBBoxes(bBoxes: List[BoundingBox]) -> List[Cuboid]:
        cuboids = []
        for bbox in bBoxes:
            a = bbox.xMax - bbox.xMin
            b = bbox.yMax - bbox.yMin
            c = bbox.zMax - bbox.zMin
            cuboids.append(Cuboid(a=a, b=b, c=c, position=bbox.center))
        return cuboids


class RandomPositionAndOrientationRaySource(RaySource):
    def __init__(self, amount, xyzLimits, position=None):
        self._position = position
        self._amount = amount
        self._limits = xyzLimits
        super(RandomPositionAndOrientationRaySource, self).__init__()

    def _createRays(self):
        if self._position is None:
            origin_xs = np.random.uniform(self._limits[0][0], self._limits[0][1], self._amount)
            origin_ys = np.random.uniform(self._limits[1][0], self._limits[1][1], self._amount)
            origin_zs = np.random.uniform(self._limits[2][0], self._limits[2][1], self._amount)
        else:
            origin_xs = np.full(self._amount, self._position.x)
            origin_ys = np.full(self._amount, self._position.y)
            origin_zs = np.full(self._amount, self._position.z)

        direction_xs = np.random.uniform(-1, 1, self._amount)
        direction_ys = np.random.uniform(-1, 1, self._amount)
        direction_zs = np.random.uniform(-1, 1, self._amount)
        for i in range(self._amount):
            self._rays.append(Ray(Vector(origin_xs[i], origin_ys[i], origin_zs[i]), Vector(direction_xs[i], direction_ys[i], direction_zs[i])))


if __name__ == '__main__':
    benchmark = IntersectionFinderBenchmark(rayAmount=10, maxDepth=25, minLeafSize=6, factor=10)
    benchmark.constructors = [NoSplitThreeAxesConstructor()]
    # benchmark.runBenchmark()
    benchmark.runValidation(resolution=25, displayFailed=True)

    #  GRAPH OF IMPROVEMENT FACTOR VS POLYGON COUNT
    # scene0 = ASphereScene(order=0)
    # scene1 = ASphereScene(order=1)
    # scene2 = ASphereScene(order=2)
    # scene3 = ASphereScene(order=3)
    # scene4 = ASphereScene(order=4)
    # scene5 = ASphereScene(order=5)
    # benchmark.scenes = [scene0, scene1, scene2, scene3, scene4, scene5]
    # benchmark.runBenchmark()
    # benchmark.displayStats()
    # benchmark.displayBenchmarkTreeResults()
