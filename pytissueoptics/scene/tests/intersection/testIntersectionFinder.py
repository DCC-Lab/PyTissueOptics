import math
import unittest

from ddt import ddt, data

from pytissueoptics.scene.solids import Sphere, Cube
from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.intersection import SimpleIntersectionFinder, Ray


@ddt
class TestIntersectionFinder(unittest.TestCase):
    intersectionFinders = [SimpleIntersectionFinder]
    intersectionFindersWithAnyPrimitives = [(finder, primitive) for finder in intersectionFinders
                                            for primitive in [primitives.TRIANGLE, primitives.QUAD]]

    @data(*intersectionFinders)
    def testGivenNoSolids_shouldNotFindIntersection(self, AnyIntersectionFinder):
        origin = Vector(0, 0, 0)
        direction = Vector(0, 0, 1)
        ray = Ray(origin, direction)

        intersection = AnyIntersectionFinder([]).findIntersection(ray)

        self.assertIsNone(intersection)

    @data(*intersectionFinders)
    def testGivenRayIsNotIntersectingASolid_shouldNotFindIntersection(self, AnyIntersectionFinder):
        direction = Vector(1, 0, 1)
        direction.normalize()
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 5))

        intersection = AnyIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNone(intersection)

    @data(*intersectionFinders)
    def testGivenRayIsIntersectingASolid_shouldReturnIntersectionDistanceAndPosition(self, AnyIntersectionFinder):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Sphere(radius=1, order=2, position=Vector(0, 0, 5))

        intersection = AnyIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(5 - math.sqrt(3) / 2, intersection.position.z, places=2)
        self.assertAlmostEqual(5 - math.sqrt(3) / 2, intersection.distance, places=2)

    @data(*intersectionFindersWithAnyPrimitives)
    def testGivenRayIsIntersectingASolid_shouldReturnIntersectionPolygon(self, anyFinderAndPrimitivePair):
        AnyIntersectionFinder, anyPrimitive = anyFinderAndPrimitivePair
        ray = Ray(origin=Vector(-0.5, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 5), primitive=anyPrimitive)
        polygonThatShouldBeHit = solid.surfaces.getPolygons("Front")[0]

        intersection = AnyIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(polygonThatShouldBeHit, intersection.polygon)

    @data(*intersectionFinders)
    def testGivenRayIsOnlyIntersectingWithASolidBoundingBox_shouldNotFindIntersection(self, AnyIntersectionFinder):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 2))

        intersection = AnyIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNone(intersection)

    @data(*intersectionFinders)
    def testGivenRayIsIntersectingMultipleSolids_shouldReturnClosestIntersection(self, AnyIntersectionFinder):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid1 = Sphere(radius=1, order=2, position=Vector(0, 0, 5))
        solid2 = Sphere(radius=1, order=1, position=Vector(0, 0, 6))
        solid3 = Sphere(radius=1, order=1, position=Vector(0, 0, 7))
        solid4 = Sphere(radius=1, order=1, position=Vector(0, 2, 3))
        solids = [solid1, solid2, solid3, solid4]

        intersection = AnyIntersectionFinder(solids).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(5 - math.sqrt(3) / 2, intersection.position.z, places=2)

    @data(*intersectionFinders)
    def testGivenRayThatFirstOnlyIntersectsWithAnotherSolidBoundingBoxBeforeIntersectingASolid_shouldFindIntersection(self, AnyIntersectionFinder):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solidMissed = Sphere(radius=1, order=1, position=Vector(0, 0, 2))
        solidHitBehind = Cube(2, position=Vector(0, 2, 4))
        solids = [solidMissed, solidHitBehind]

        intersection = AnyIntersectionFinder(solids).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.9*3, intersection.position.y)
        self.assertEqual(3, intersection.position.z)
