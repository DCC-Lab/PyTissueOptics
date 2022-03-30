import unittest

from pytissueoptics.scene.geometry import Triangle, Vector, Quad, Polygon
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.mollerTrumboreIntersect import MollerTrumboreIntersect


class TestAnyPolygonIntersect(unittest.TestCase):
    vertices = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0), Vector(0, 1, 0)]
    triangle = Triangle(vertices[0], vertices[1], vertices[3])
    quad = Quad(vertices[0], vertices[1], vertices[2], vertices[3])
    polygon = Polygon(vertices=[vertices[0], vertices[1], vertices[2], vertices[3]])

    def setUp(self):
        self.intersectStrategy = MollerTrumboreIntersect()

    def testGivenIntersectingRayAndPolygon_shouldReturnIntersectionPosition(self):
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        for poly in ([self.triangle, self.quad, self.polygon]):
            intersection = self.intersectStrategy.getIntersection(ray, poly)
            self.assertEqual(0.45, intersection.x)
            self.assertEqual(0.25, intersection.y)
            self.assertEqual(0.0, intersection.z)

    def testGivenNonIntersectingRayAndPolygon_shouldReturnNone(self):
        rayOrigin = Vector(0.25, 0.25, 1)
        rayDirection = Vector(-0.3, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        for poly in ([self.triangle, self.quad, self.polygon]):
            intersection = self.intersectStrategy.getIntersection(ray, poly)
            self.assertIsNone(intersection)

    def testGivenLineIntersectingRayAndPolygon_shouldReturnNone(self):
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        for poly in ([self.triangle, self.quad, self.polygon]):
            intersection = self.intersectStrategy.getIntersection(ray, poly)
            self.assertIsNone(intersection)

    def testGivenRayShorterThanPolygonIntersectionDistance_shouldReturnNone(self):
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=1.8)

        for poly in ([self.triangle, self.quad, self.polygon]):
            intersection = self.intersectStrategy.getIntersection(ray, poly)
            self.assertIsNone(intersection)

    def testGivenRayLongerThanPolygonIntersectionDistance_shouldReturnIntersectionPosition(self):
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=2.2)

        for poly in ([self.triangle, self.quad, self.polygon]):
            intersection = self.intersectStrategy.getIntersection(ray, poly)

            self.assertIsNotNone(intersection)
            self.assertEqual(0.45, intersection.x)
            self.assertEqual(0.25, intersection.y)
            self.assertEqual(0.0, intersection.z)
