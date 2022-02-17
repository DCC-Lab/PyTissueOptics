import unittest

from ddt import ddt, data

from pytissueoptics.scene.geometry import Triangle, Vector
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreIntersect


@ddt
class TestTriangleIntersect(unittest.TestCase):
    intersectStrategies = [MollerTrumboreIntersect]

    @data(*intersectStrategies)
    def testGivenIntersectingRayAndTriangle_shouldReturnIntersectionPoint(self, IntersectStrategy):
        triangle = Triangle(Vector(0 + 2, 0, 0), Vector(1 + 2, 0, 0), Vector(0 + 2, 1, 0))
        rayOrigin = Vector(0.25 + 2, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, triangle)

        self.assertEqual(2.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)

    @data(*intersectStrategies)
    def testGivenNonIntersectingRayAndTriangle_shouldReturnNone(self, IntersectStrategy):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
        rayOrigin = Vector(0.25, 0.25, 1)
        rayDirection = Vector(-0.3, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, triangle)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenLineIntersectingRayAndTriangle_shouldReturnNone(self, IntersectStrategy):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = IntersectStrategy().getIntersection(ray, triangle)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayShorterThanTriangleIntersection_shouldReturnNone(self, IntersectStrategy):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=1.8)

        intersection = IntersectStrategy().getIntersection(ray, triangle)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayShorterThanTriangleIntersection_shouldReturnNone(self, IntersectStrategy):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=2.2)

        intersection = IntersectStrategy().getIntersection(ray, triangle)

        self.assertIsNotNone(intersection)
        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)
