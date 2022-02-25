import unittest

from ddt import ddt, data

from pytissueoptics.scene.geometry import Triangle, Vector, Quad
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.quadIntersect import MollerTrumboreQuadIntersect
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreTriangleIntersect


@ddt
class TestPolygonIntersect(unittest.TestCase):
    intersectStrategies = [(MollerTrumboreTriangleIntersect,
                            Triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))),
                           (MollerTrumboreQuadIntersect,
                            Quad(Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0), Vector(0, 1, 0)))]

    @data(*intersectStrategies)
    def testGivenIntersectingRayAndPolygon_shouldReturnIntersectionPosition(self, anyStrategyAndPolygonPair):
        AnyIntersectStrategy, polygon = anyStrategyAndPolygonPair
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = AnyIntersectStrategy().getIntersection(ray, polygon)

        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)

    @data(*intersectStrategies)
    def testGivenNonIntersectingRayAndPolygon_shouldReturnNone(self, anyStrategyAndPolygonPair):
        AnyIntersectStrategy, polygon = anyStrategyAndPolygonPair
        rayOrigin = Vector(0.25, 0.25, 1)
        rayDirection = Vector(-0.3, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = AnyIntersectStrategy().getIntersection(ray, polygon)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenLineIntersectingRayAndPolygon_shouldReturnNone(self, anyStrategyAndPolygonPair):
        AnyIntersectStrategy, polygon = anyStrategyAndPolygonPair
        rayOrigin = Vector(-1, -1, 0)
        rayDirection = Vector(1, 1, 0)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection)

        intersection = AnyIntersectStrategy().getIntersection(ray, polygon)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayShorterThanPolygonIntersectionDistance_shouldReturnNone(self, anyStrategyAndPolygonPair):
        AnyIntersectStrategy, polygon = anyStrategyAndPolygonPair
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=1.8)

        intersection = AnyIntersectStrategy().getIntersection(ray, polygon)

        self.assertIsNone(intersection)

    @data(*intersectStrategies)
    def testGivenRayLongerThanPolygonIntersectionDistance_shouldReturnIntersectionPosition(self, anyStrategyAndPolygonPair):
        AnyIntersectStrategy, polygon = anyStrategyAndPolygonPair
        rayOrigin = Vector(0.25, 0.25, 2)
        rayDirection = Vector(0.1, 0, -1)
        rayDirection.normalize()
        ray = Ray(rayOrigin, rayDirection, length=2.2)

        intersection = AnyIntersectStrategy().getIntersection(ray, polygon)

        self.assertIsNotNone(intersection)
        self.assertEqual(0.45, intersection.x)
        self.assertEqual(0.25, intersection.y)
        self.assertEqual(0.0, intersection.z)
