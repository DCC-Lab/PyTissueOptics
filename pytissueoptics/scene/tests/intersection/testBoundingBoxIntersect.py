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
        box = BoundingBox([0+2, 1+2], [0, 1], [-1, 0])
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
    def testGivenRayInsideBox_shouldRaiseNotImplementedError(self, IntersectStrategy):
        box = BoundingBox([0, 1], [0, 1], [-2, 1])
        rayOrigin = Vector(0.25, 0.25, 0)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        with self.assertRaises(NotImplementedError):
            _ = IntersectStrategy().getIntersection(ray, box)
