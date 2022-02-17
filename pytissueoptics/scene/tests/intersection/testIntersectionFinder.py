import math
import unittest

from ddt import ddt, data

from pytissueoptics.scene.solids import Sphere
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.intersection import SimpleIntersectionFinder, Ray


@ddt
class TestIntersectionFinder(unittest.TestCase):
    intersectionFinders = [SimpleIntersectionFinder]

    @data(*intersectionFinders)
    def testGivenNoSolids_shouldNotFindIntersection(self, IntersectionFinder):
        intersectionFinder = IntersectionFinder([])
        origin = Vector(0, 0, 0)
        direction = Vector(0, 0, 1)
        ray = Ray(origin, direction)

        intersection = intersectionFinder.findIntersection(ray)

        self.assertIsNone(intersection)

    @data(*intersectionFinders)
    def testGivenRayIsNotIntersectingASolid_shouldNotFindIntersection(self, IntersectionFinder):
        direction = Vector(1, 0, 1)
        direction.normalize()
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 5))
        intersectionFinder = IntersectionFinder([solid])

        intersection = intersectionFinder.findIntersection(ray)

        self.assertIsNone(intersection)

    @data(*intersectionFinders)
    def testGivenRayIsIntersectingASolid_shouldReturnIntersection(self, IntersectionFinder):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Sphere(radius=1, order=2, position=Vector(0, 0, 5))
        intersectionFinder = IntersectionFinder([solid])

        intersection = intersectionFinder.findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(5 - math.sqrt(3)/2, intersection.position.z, places=2)

    @data(*intersectionFinders)
    def testGivenRayIsOnlyIntersectingWithASolidBoundingBox_shouldNotFindIntersection(self, IntersectionFinder):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 2))
        intersectionFinder = IntersectionFinder([solid])

        intersection = intersectionFinder.findIntersection(ray)

        self.assertIsNone(intersection)

    @data(*intersectionFinders)
    def testGivenRayIsIntersectingMultipleSolids_shouldReturnClosestIntersection(self, IntersectionFinder):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid1 = Sphere(radius=1, order=2, position=Vector(0, 0, 5))
        solid2 = Sphere(radius=1, order=1, position=Vector(0, 0, 6))
        solid3 = Sphere(radius=1, order=1, position=Vector(0, 0, 7))
        solid4 = Sphere(radius=1, order=1, position=Vector(0, 2, 3))

        intersectionFinder = IntersectionFinder([solid1, solid2, solid3, solid4])

        intersection = intersectionFinder.findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(5 - math.sqrt(3)/2, intersection.position.z, places=2)
