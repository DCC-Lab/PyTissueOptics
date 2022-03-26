from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.modernKDTreeConstructor import ModernKDTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.threeAxesNoSplitTreeConstructor import \
    ThreeAxesNoSplitTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.threeAxesSplitTreeConstructor import ThreeAxesSplitTreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary.oneAxisNoSplitTreeConstructor import OneAxisNoSplitTreeConstructor
from pytissueoptics.scene.intersection import FastIntersectionFinder, SimpleIntersectionFinder, Ray, UniformRaySource, \
    RandomPositionAndOrientationRaySource, RaySource

from pytissueoptics.scene.tests.scene.benchmarkScenes import *
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.intersection.intersectionFinder import Intersection
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.viewer import MayaviViewer
import pandas
import time

pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 1200)
RPORaySource = RandomPositionAndOrientationRaySource


class IntersectionFinderBenchmark:
    def __init__(self, rayAmount=10000, maxDepth=100, minLeafSize=0, factor=10, constructors=None, displayViewer=False):
        self.scenes = self._getScenes()
        if constructors is None:
            self.constructors = [OneAxisNoSplitTreeConstructor(), ThreeAxesNoSplitTreeConstructor(),
                                 ThreeAxesSplitTreeConstructor(), ModernKDTreeConstructor()]
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

    def runValidation(self):
        print(f"=================== VALIDATION ====================")
        self._launchValidationReference()
        for constructor in self.constructors:
            self._launchValidationForConstructor(constructor)

    def _launchValidationReference(self):
        logger = Logger()
        scene = self.scenes[7]
        source = UniformRaySource(position=Vector(0, 4, 0), direction=Vector(0, 0, -1), xTheta=360, yTheta=90,
                                  xResolution=512, yResolution=128)
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
            f" - {0.00:.2f}s"
            f" - {traversalTime:.2f}s"
            f" - {missedRays}"
            f" - PASSED: {missedRays == 25552}")
        viewer = MayaviViewer()
        viewer.addLogger(logger)
        viewer.show()

    def _launchValidationForConstructor(self, constructor):
        logger = Logger()
        scene = self.scenes[7]
        source = UniformRaySource(position=Vector(0, 4, 0), direction=Vector(0, 0, -1), xTheta=360, yTheta=90,
                                  xResolution=512, yResolution=128)
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
        print(
            f"{constructor.__class__.__name__:^20.20s}"
            f" - {constructionTime:.2f}s"
            f" - {traversalTime:.2f}s"
            f" - {missedRays}"
            f" - PASSED: {missedRays == 25552}")
        viewer = MayaviViewer()
        viewer.addLogger(logger)
        viewer.show()

    def run(self):
        print(f"=================== BENCHMARK ====================")
        for scene in self.scenes:
            self.launchBenchmarkForScene(scene)

    def launchBenchmarkForScene(self, scene: Scene):
        print(f"\n================ {scene.__class__.__name__:.15s} =================")
        print(f"Scene polygonCount: {len(scene.getPolygons())}, Scene Object Count: {len(scene.solids)}")
        print(f"================ {scene.__class__.__name__:.15s} =================")
        self.partitions.append([])
        self.launchReferenceBenchmarkForScene(scene)
        self.launchBenchmarkForSceneWithSource(scene, RPORaySource(self.rayAmount, scene.getBoundingBox().xyzLimits))

    def launchReferenceBenchmarkForScene(self, scene: Scene):
        self.count += 1
        source = RandomPositionAndOrientationRaySource(int(self.rayAmount / self.factor),
                                                       scene.getBoundingBox().xyzLimits)
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

    def launchBenchmarkForSceneWithSource(self, scene: Scene, source: RaySource):
        for constructor in self.constructors:
            self.launchBenchmarkForSceneWithConstructor(scene, constructor, source)

    def launchBenchmarkForSceneWithConstructor(self, scene: Scene, constructor: TreeConstructor, source: RaySource):
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
                                               f"-",
                                               f"{intersectionFinder.__class__.__name__:^12.15s}",
                                               f"-",
                                               f"{traversalTime:^12.2f}",
                                               f"{traversalTime:^12.2f}",
                                               f"-",
                                               f"-",
                                               f"-",
                                               f"-",
                                               f"-"]

    def _saveFastStats(self, scene: Scene, intersectionFinder: FastIntersectionFinder, traversalTime: float,
                       buildTime: float):
        partition = intersectionFinder.partition
        self.partitions[-1].append(partition)
        self.stats.loc[self.stats.shape[0]] = [f"{scene.__class__.__name__}", f"{len(scene.getPolygons()):^12}",
                                               f"{len(partition.getLeafPolygons()):^12}",
                                               f"{partition._constructor.__class__.__name__:^12.15s}",
                                               f"{buildTime:^12.2f}",
                                               f"{traversalTime:^12.2f}",
                                               f"{buildTime + traversalTime:^12.2f}",
                                               f"{partition.getNodeCount():^12}",
                                               f"{partition.getLeafCount():^12}",
                                               f"{partition.getAverageLeafDepth():^12.2f}",
                                               f"{partition.getAverageLeafSize():^12.2f}",
                                               f"{((self.simpleTraversalTime[-1]) / traversalTime):.1f}"]

    @staticmethod
    def _measureSignal(intersection: Intersection) -> float:
        if intersection.polygon.insideMaterial is None:
            return 0.125
        return 1.0

    def displayStats(self):
        print(self.stats)

    def displayBenchmarkTreeResults(self):
        self.viewer = MayaviViewer()
        for j, scene in enumerate(self.scenes):
            for partition in self.partitions[j]:
                bBoxes = partition.getLeafBoundingBoxesAsCuboids()
                self.viewer = MayaviViewer()
                self.viewer.add(*scene.getSolids(), representation="surface", lineWidth=0.05, opacity=0.5)
                self.viewer.add(*bBoxes, representation="wireframe", lineWidth=3, color=(1, 0, 0), opacity=0.7)
                self.viewer.show()
                self.viewer.clear()


if __name__ == '__main__':
    benchmark = IntersectionFinderBenchmark()
    benchmark.runValidation()
