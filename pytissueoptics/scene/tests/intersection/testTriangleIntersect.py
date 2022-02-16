import unittest

from pytissueoptics.scene.geometry import Triangle, Vector
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreIntersect


class TestMollerTrumboreIntersect(unittest.TestCase):
    def testGivenIntersectingRayAndTriangle_shouldReturnIntersectionPoint(self):
        triangle = Triangle(Vector(0+2, 0, 0), Vector(1+2, 0, 0), Vector(0+2, 1, 0))
        rayOrigin = Vector(0.25+2, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        ray = Ray(rayOrigin, rayDirection)

        intersection = MollerTrumboreIntersect().getIntersection(ray, triangle)

        self.assertEqual(2.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)

    def testGivenNonIntersectingRayAndTriangle_shouldReturnNone(self):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
        rayOrigin = Vector(0.25, 0.25, 1)
        rayDirection = Vector(-0.3, 0, -1)
        ray = Ray(rayOrigin, rayDirection)

        intersection = MollerTrumboreIntersect().getIntersection(ray, triangle)

        self.assertIsNone(intersection)

    def testGivenLineIntersectingRayAndTriangle_shouldReturnNone(self):
        triangle = Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        ray = Ray(rayOrigin, rayDirection)

        intersection = MollerTrumboreIntersect().getIntersection(ray, triangle)

        self.assertIsNone(intersection)
