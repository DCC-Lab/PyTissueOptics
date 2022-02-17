import unittest

from ddt import ddt, data

from pytissueoptics.scene.geometry import Vector, BoundingBox
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.bboxIntersect import GemsBoxIntersect, ZacharBoxIntersect


@ddt
class TestBoxIntersect(unittest.TestCase):
    intersectStrategies = [GemsBoxIntersect, ZacharBoxIntersect]

    @data(*intersectStrategies)
    def testGivenIntersectingRayAndBox_shouldReturnClosestIntersectionPoint(self, IntersectStrategy):
        box = BoundingBox([0 + 2, 1 + 2], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25 + 2, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, box)

        self.assertIsNotNone(intersection)
        self.assertEqual(2.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)

    @data(*intersectStrategies)
    def testGivenNonIntersectingRayAndBox_shouldReturnNone(self, IntersectStrategy):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(-0.2, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, box)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayPointingAwayFromTheBox_shouldReturnNone(self, IntersectStrategy):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(-0.1, 0, 1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, box)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayInsideBox_shouldReturnRayOrigin(self, IntersectStrategy):
        box = BoundingBox([0, 1], [0, 1], [-2, 1])
        rayOrigin = Vector(0.25, 0.25, 0)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, box)
        
        self.assertEqual(ray.origin, intersection)

    @data(*intersectStrategies)
    def testGivenRayLengthShorterThanBoxIntersection_shouldReturnNone(self, IntersectStrategy):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=1.8)

        intersection = IntersectStrategy().getIntersection(ray, box)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayLengthLongerThanBoxIntersection_shouldReturnIntersection(self, IntersectStrategy):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=2.2)

        intersection = IntersectStrategy().getIntersection(ray, box)

        self.assertIsNotNone(intersection)
        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)


class TestGemsBoxIntersect(unittest.TestCase):
    def testGivenLineIntersectingRayAndBox_shouldReturnClosestIntersectionPoint(self):
        box = BoundingBox([1, 2], [1, 2], [-1, 0])
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = GemsBoxIntersect().getIntersection(ray, box)

        self.assertIsNotNone(intersection)
        self.assertEqual(1, intersection.x)
        self.assertEqual(1, intersection.y)
        self.assertEqual(0, intersection.z)


class TestZacharBoxIntersect(unittest.TestCase):
    def testGivenLineIntersectingRayAndBox_shouldReturnNone(self):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = ZacharBoxIntersect().getIntersection(ray, box)

        self.assertIsNone(intersection)
