import math
import unittest

from pytissueoptics.scene.intersection.intersectionFinder import IntersectionFinder
from pytissueoptics.scene.solids import Sphere, Cube, Cuboid
from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.intersection import SimpleIntersectionFinder, FastIntersectionFinder, Ray


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


class TestAnyIntersectionFinder:

    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        raise NotImplementedError

    def testGivenNoSolids_shouldNotFindIntersection(self):
        origin = Vector(0, 0, 0)
        direction = Vector(0, 0, 1)
        ray = Ray(origin, direction)

        intersection = self.getIntersectionFinder([]).findIntersection(ray)

        self.assertIsNone(intersection)

    def testGivenRayIsNotIntersectingASolid_shouldNotFindIntersection(self):
        direction = Vector(1, 0, 1)
        direction.normalize()
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNone(intersection)

    def testGivenRayIsIntersectingASolid_shouldReturnIntersectionDistanceAndPosition(self):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Sphere(radius=1, order=2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(5 - math.sqrt(3) / 2, intersection.position.z, places=2)
        self.assertAlmostEqual(5 - math.sqrt(3) / 2, intersection.distance, places=2)

    def testGivenRayIsIntersectingASolidWithTrianglePrimitive_shouldReturnIntersectionTriangle(self):
        self._testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygon(primitives.TRIANGLE)

    def testGivenRayIsIntersectingASolidWithQuadPrimitive_shouldReturnIntersectionQuad(self):
        self._testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygon(primitives.QUAD)

    def _testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygon(self, anyPrimitive):
        ray = Ray(origin=Vector(-0.5, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 5), primitive=anyPrimitive)
        polygonThatShouldBeHit = solid.surfaces.getPolygons("Front")[0]

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(polygonThatShouldBeHit, intersection.polygon)

    def testGivenRayIsOnlyIntersectingWithASolidBoundingBox_shouldNotFindIntersection(self):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 2))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNone(intersection)

    def testGivenRayIsIntersectingMultipleSolids_shouldReturnClosestIntersection(self):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid1 = Sphere(radius=1, order=2, position=Vector(0, 0, 5))
        solid2 = Sphere(radius=1, order=1, position=Vector(0, 0, 8))
        solid3 = Sphere(radius=1, order=1, position=Vector(0, 0, 11))
        solids = [solid1, solid2, solid3]


        intersection = self.getIntersectionFinder(solids).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(5 - math.sqrt(3) / 2, intersection.position.z, places=2)

    def testGivenRayThatFirstOnlyIntersectsWithAnotherSolidBoundingBoxBeforeIntersectingASolid_shouldFindIntersection(self):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solidMissed = Sphere(radius=1, order=1, position=Vector(0, 0, 1.9))
        solidHitBehind = Cube(2, position=Vector(0, 2, 4))
        solids = [solidMissed, solidHitBehind]

        intersection = self.getIntersectionFinder(solids).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.9*3, intersection.position.y)
        self.assertEqual(3, intersection.position.z)


class TestSimpleIntersectionFinder(TestAnyIntersectionFinder, unittest.TestCase):
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        scene = Scene(solids)
        return SimpleIntersectionFinder(scene)


class TestFastIntersectionFinder(TestAnyIntersectionFinder, unittest.TestCase):
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        scene = Scene(solids)
        return FastIntersectionFinder(scene)


class TestEndToEndIntersection(unittest.TestCase):

    def setUp(self) -> None:
        scene = PhantomScene()
        self.intersectionFinder = FastIntersectionFinder(scene)
    
    def testGivenRayTowardsBackWall(self):
        origin = Vector(0, 4, 0)
        direction = Vector(0, 0, -1)
        ray = Ray(origin, direction)
        intersection = self.intersectionFinder.findIntersection(ray)
        print(intersection)
