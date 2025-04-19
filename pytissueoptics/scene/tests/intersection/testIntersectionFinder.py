import math
import unittest

from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.geometry.polygon import WORLD_LABEL
from pytissueoptics.scene.intersection import FastIntersectionFinder, Ray, SimpleIntersectionFinder, UniformRaySource
from pytissueoptics.scene.intersection.intersectionFinder import IntersectionFinder
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.solids import Cube, Sphere
from pytissueoptics.scene.tests.scene.benchmarkScenes import PhantomScene
from pytissueoptics.scene.tree.treeConstructor.binary import (
    NoSplitOneAxisConstructor,
    NoSplitThreeAxesConstructor,
    SplitThreeAxesConstructor,
)


class BaseTestAnyIntersectionFinder:
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        raise NotImplementedError

    def testGivenNoSolids_shouldNotFindIntersection(self):
        origin = Vector(0, 0, 0)
        direction = Vector(0, 0, 1)
        ray = Ray(origin, direction)

        intersection = self.getIntersectionFinder([]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNone(intersection)

    def testGivenRayIsNotIntersectingASolid_shouldNotFindIntersection(self):
        direction = Vector(1, 0, 1)
        direction.normalize()
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Cube(2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNone(intersection)

    def testGivenRayIsIntersectingASolid_shouldReturnIntersectionDistanceAndPosition(self):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(4, intersection.position.z)
        self.assertAlmostEqual(4, intersection.distance)

    def testGivenRayIsIntersectingASolidWithTrianglePrimitive_shouldReturnIntersectionTriangleNormal(self):
        self._testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygonNormal(
            primitives.TRIANGLE
        )

    def testGivenRayIsIntersectingASolidWithQuadPrimitive_shouldReturnIntersectionQuadNormal(self):
        self._testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygonNormal(primitives.QUAD)

    def _testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygonNormal(self, anyPrimitive):
        ray = Ray(origin=Vector(-0.5, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 5), primitive=anyPrimitive)
        polygonThatShouldBeHit = solid.surfaces.getPolygons("front")[0]

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNotNone(intersection)
        self.assertEqual(polygonThatShouldBeHit.normal, intersection.normal)

    def testGivenRayIsOnlyIntersectingWithASolidBoundingBox_shouldNotFindIntersection(self):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 2))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNone(intersection)

    def testGivenRayIsIntersectingMultipleSolids_shouldReturnClosestIntersection(self):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid1 = Cube(2, position=Vector(0, 0, 5), label="solid1")
        solid2 = Cube(2, position=Vector(0, 0, 10), label="solid2")
        solids = [solid1, solid2]

        intersection = self.getIntersectionFinder(solids).findIntersection(ray, WORLD_LABEL)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(4, intersection.position.z)

    def testGivenRayThatFirstOnlyIntersectsWithAnotherSolidBoundingBoxBeforeIntersectingASolid_shouldFindIntersection(self):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solidMissed = Sphere(radius=1, order=1, position=Vector(0, 0, 1.9))
        solidHitBehind = Cube(2, position=Vector(0, 2, 4))
        solids = [solidMissed, solidHitBehind]

        intersection = self.getIntersectionFinder(solids).findIntersection(ray, WORLD_LABEL)

        self.assertIsNotNone(intersection)
        self.assertAlmostEqual(0, intersection.position.x, 4)
        self.assertAlmostEqual(0.9 * 3, intersection.position.y, 4)
        self.assertAlmostEqual(3, intersection.position.z, 4)

    def testGivenRayIsOnACubeSurface_shouldFindIntersectionAtCrossing(self):
        ray = Ray(origin=Vector(0, 1, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, solid.getLabel())

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(1, intersection.position.y)
        self.assertEqual(1, intersection.position.z)

    def testGivenRayIsOnACubeEdge_shouldFindIntersectionAtCrossing(self):
        ray = Ray(origin=Vector(1, 1, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, solid.getLabel())

        self.assertIsNotNone(intersection)
        self.assertEqual(1, intersection.position.x)
        self.assertEqual(1, intersection.position.y)
        self.assertEqual(1, intersection.position.z)

    def testGivenOutsideRayIsOnACubeEdge_shouldNotFindIntersection(self):
        ray = Ray(origin=Vector(1, 1, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNone(intersection)

    def testGivenRayStartsAtAnInterface_shouldFindIntersectionAtRayOrigin(self):
        ray = Ray(origin=Vector(0, 0, 1), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, solid.getLabel())

        self.assertIsNotNone(intersection)
        self.assertVectorEqual(ray.origin, intersection.position)

    def testGivenRayStartsAtACubeCorner_shouldFindIntersectionAtRayOrigin(self):
        ray = Ray(origin=Vector(1, 1, 1), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, solid.getLabel())

        self.assertIsNotNone(intersection)
        self.assertVectorEqual(ray.origin, intersection.position)

    def testGivenRayStartsSlightlyInsideAnInterfaceAndLeaves_shouldFindIntersection(self):
        eps = 1e-6  # 1e-7 will not pass
        ray = Ray(origin=Vector(0, 0, 1 - eps), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, solid.getLabel())

        self.assertIsNotNone(intersection)

    def testGivenRayStartsSlightlyInsideCubeCornerAndLeaves_shouldFindIntersection(self):
        eps = 1e-6  # 1e-7 will not pass
        ray = Ray(origin=Vector(1 - eps, 1 - eps, 1 - eps), direction=Vector(1, 1, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, solid.getLabel())

        self.assertIsNotNone(intersection)
        self.assertEqual(1, intersection.position.x)
        self.assertEqual(1, intersection.position.y)
        self.assertEqual(1, intersection.position.z)

    def testGivenOutsideRayStartsSlightlyInsideCubeCornerAndLeaves_shouldNotFindIntersection(self):
        eps = 1e-6
        ray = Ray(origin=Vector(1 - eps, 1 - eps, 1 - eps), direction=Vector(1, 1, 1))
        solid = Cube(2, position=Vector(0, 0, 0))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNone(intersection)

    def testGivenRayLengthIsLongerThanDistanceToIntersection_shouldFindIntersection(self):
        ray = Ray(origin=Vector(0, 0, 0), direction=Vector(0, 0, 1), length=10)
        solid = Cube(2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0, intersection.position.y)
        self.assertEqual(4, intersection.position.z)

    def testGivenRayLengthIsShorterThanDistanceToIntersection_shouldNotFindIntersection(self):
        ray = Ray(origin=Vector(0, 0, 0), direction=Vector(0, 0, 1), length=3)
        solid = Cube(2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        self.assertIsNone(intersection)

    def testGivenSmoothSolid_shouldSmoothTheIntersectionNormal(self):
        ray = Ray(origin=Vector(0, 0.7, -3), direction=Vector(0, 0, 1))
        solid = Sphere(radius=1, order=1)

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        expectedAngle = math.atan(intersection.position.y / intersection.position.z)
        expectedNormal = Vector(0, -math.sin(expectedAngle), -math.cos(expectedAngle))
        self.assertIsNotNone(intersection)
        self.assertEqual(expectedNormal, intersection.normal)

    def testGivenSmoothSolid_shouldNotSmoothTheIntersectionNormalIfItChangesTheSignOfTheDotProductWithTheRayDirection(self):
        ray = Ray(origin=Vector(0, 0.955, -2), direction=Vector(0, 0.02, 1))
        solid = Sphere(radius=1, order=1)

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray, WORLD_LABEL)

        smoothAngle = math.atan(intersection.position.y / intersection.position.z)
        smoothNormal = Vector(0, -math.sin(smoothAngle), -math.cos(smoothAngle))
        self.assertIsNotNone(intersection)
        self.assertNotEqual(smoothNormal, intersection.normal)

    def assertVectorEqual(self, expected, actual):
        self.assertEqual(expected.x, actual.x)
        self.assertEqual(expected.y, actual.y)
        self.assertEqual(expected.z, actual.z)


class TestSimpleIntersectionFinder(BaseTestAnyIntersectionFinder, unittest.TestCase):
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        scene = Scene(solids)
        return SimpleIntersectionFinder(scene)


class TestFastIntersectionFinder(BaseTestAnyIntersectionFinder, unittest.TestCase):
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        scene = Scene(solids)
        return FastIntersectionFinder(scene)


class TestEndToEndIntersection(unittest.TestCase):
    def setUp(self) -> None:
        scene = PhantomScene()
        self.intersectionFinders = [
            SimpleIntersectionFinder(scene),
            FastIntersectionFinder(scene, constructor=NoSplitOneAxisConstructor(), maxDepth=3),
            FastIntersectionFinder(scene, constructor=NoSplitThreeAxesConstructor(), maxDepth=3),
            FastIntersectionFinder(scene, constructor=SplitThreeAxesConstructor(), maxDepth=3)
        ]

    def _getSubTestTag(self, intersectionFinder: IntersectionFinder):
        if isinstance(intersectionFinder, FastIntersectionFinder):
            return intersectionFinder._partition._constructor.__class__.__name__
        return intersectionFinder.__class__.__name__

    def testGivenRayTowardsBackWall_shouldReturnCorrectIntersection(self):
        origin = Vector(0, 4, 0)
        direction = Vector(0, 0, -1)
        ray = Ray(origin, direction)
        for intersectionFinder in self.intersectionFinders:
            with self.subTest(self._getSubTestTag(intersectionFinder)):
                intersection = intersectionFinder.findIntersection(ray, WORLD_LABEL)
                expectedPosition = Vector(0, 4, -9.95)
                self.assertEqual(expectedPosition, intersection.position)

    def testGivenRaysTowardsScene_shouldNeverReturnNone(self):
        rays = UniformRaySource(Vector(0, 4, 0), Vector(0, 0, -1), 180, 0, xResolution=20, yResolution=1).rays
        for intersectionFinder in self.intersectionFinders:
            with self.subTest(self._getSubTestTag(intersectionFinder)):
                for ray in rays:
                    intersection = intersectionFinder.findIntersection(ray, WORLD_LABEL)
                    self.assertIsNotNone(intersection)
