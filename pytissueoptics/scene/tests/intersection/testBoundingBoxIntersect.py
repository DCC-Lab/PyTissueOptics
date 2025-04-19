import unittest

from pytissueoptics.scene.geometry import Vector, BoundingBox
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.bboxIntersect import BoxIntersectStrategy, GemsBoxIntersect, ZacharBoxIntersect


class BaseTestAnyBoxIntersect:
    @property
    def intersectStrategy(self) -> BoxIntersectStrategy:
        raise NotImplementedError

    def testGivenIntersectingRayAndBox_shouldReturnClosestIntersectionPoint(self):
        box = BoundingBox([0 + 2, 1 + 2], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25 + 2, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNotNone(intersection)
        self.assertEqual(2.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)

    def testGivenNonIntersectingRayAndBox_shouldReturnNone(self):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(-0.2, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNone(intersection)

    def testGivenRayPointingAwayFromTheBox_shouldReturnNone(self):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(-0.1, 0, 1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNone(intersection)

    def testGivenRayInsideBox_shouldReturnRayOrigin(self):
        box = BoundingBox([0, 1], [0, 1], [-2, 1])
        rayOrigin = Vector(0.25, 0.25, 0)
        rayDirection = Vector(0, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, box)
        
        self.assertEqual(ray.origin, intersection)

    def testGivenRayLengthShorterThanBoxIntersection_shouldReturnNone(self):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=1.8)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNone(intersection)

    def testGivenRayLengthLongerThanBoxIntersection_shouldReturnIntersection(self):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=2.2)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNotNone(intersection)
        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)


class TestGemsBoxIntersect(BaseTestAnyBoxIntersect, unittest.TestCase):
    def testGivenLineIntersectingRayAndBox_shouldReturnClosestIntersectionPoint(self):
        box = BoundingBox([1, 2], [1, 2], [-1, 0])
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNotNone(intersection)
        self.assertEqual(1, intersection.x)
        self.assertEqual(1, intersection.y)
        self.assertEqual(0, intersection.z)

    @property
    def intersectStrategy(self) -> BoxIntersectStrategy:
        return GemsBoxIntersect()


class TestZacharBoxIntersect(BaseTestAnyBoxIntersect, unittest.TestCase):
    def testGivenLineIntersectingRayAndBox_shouldReturnNone(self):
        box = BoundingBox([0, 1], [0, 1], [-1, 0])
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, box)

        self.assertIsNone(intersection)

    @property
    def intersectStrategy(self) -> BoxIntersectStrategy:
        return ZacharBoxIntersect()
