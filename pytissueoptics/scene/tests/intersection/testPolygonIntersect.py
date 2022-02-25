import unittest

from pytissueoptics.scene.geometry import Triangle, Vector, Quad, Polygon
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.quadIntersect import MollerTrumboreQuadIntersect
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreTriangleIntersect


class TestAnyPolygonIntersect:
    vertices = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0), Vector(0, 1, 0)]

    @property
    def intersectStrategy(self):
        # todo: we need a IntersectStrategy interface before box and polygon (or triangle, quad) intersect
        raise NotImplementedError

    @property
    def aPolygon(self) -> Polygon:
        raise NotImplementedError

    def testGivenIntersectingRayAndPolygon_shouldReturnIntersectionPosition(self):
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, self.aPolygon)

        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)

    def testGivenNonIntersectingRayAndPolygon_shouldReturnNone(self):
        rayOrigin = Vector(0.25, 0.25, 1)
        rayDirection = Vector(-0.3, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, self.aPolygon)

        self.assertIsNone(intersection)

    def testGivenLineIntersectingRayAndPolygon_shouldReturnNone(self):
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = self.intersectStrategy.getIntersection(ray, self.aPolygon)

        self.assertIsNone(intersection)

    def testGivenRayShorterThanPolygonIntersectionDistance_shouldReturnNone(self):
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=1.8)

        intersection = self.intersectStrategy.getIntersection(ray, self.aPolygon)

        self.assertIsNone(intersection)

    def testGivenRayLongerThanPolygonIntersectionDistance_shouldReturnIntersectionPosition(self):
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=2.2)

        intersection = self.intersectStrategy.getIntersection(ray, self.aPolygon)

        self.assertIsNotNone(intersection)
        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)


class TestMollerTrumboreTriangleIntersect(TestAnyPolygonIntersect, unittest.TestCase):
    @property
    def intersectStrategy(self):
        return MollerTrumboreTriangleIntersect()

    @property
    def aPolygon(self):
        return Triangle(self.vertices[0], self.vertices[1], self.vertices[3])


class TestMollerTrumboreQuadIntersect(TestAnyPolygonIntersect, unittest.TestCase):
    @property
    def intersectStrategy(self):
        return MollerTrumboreQuadIntersect()

    @property
    def aPolygon(self) -> Polygon:
        return Quad(*self.vertices)
