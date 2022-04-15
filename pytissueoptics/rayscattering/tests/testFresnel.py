import math
import unittest

from pytissueoptics.rayscattering.fresnel import FresnelIntersect
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Vector
from pytissueoptics.scene.geometry import Polygon, Environment
from pytissueoptics.scene.intersection.intersectionFinder import Intersection


class TestFresnelIntersect(unittest.TestCase):
    def setUp(self):
        self.rayAt45 = Vector(1, 0, -1)
        self.rayAt45.normalize()
        self.fresnelIntersect = FresnelIntersect()

    def testWhenCompute_shouldReturnAFresnelIntersection(self):
        intersection = self._createIntersection(n1=1.0, n2=1.5)

        fresnelIntersection = self.fresnelIntersect.compute(self.rayAt45, intersection)

        self.assertIsNotNone(fresnelIntersection)
        self.assertNotEqual(0, fresnelIntersection.angleDeflection)
        self.assertEqual(Vector(0, 1, 0), fresnelIntersection.incidencePlane)

    def testWhenIsReflected_shouldComputeReflectionDeflection(self):
        intersection = self._createIntersection(n1=1.0, n2=1.5)
        self.fresnelIntersect._getIsReflected = lambda: True

        fresnelIntersection = self.fresnelIntersect.compute(self.rayAt45, intersection)

        self.assertTrue(fresnelIntersection.isReflected)
        self.assertAlmostEqual(-math.pi / 2, fresnelIntersection.angleDeflection)

    def testWhenIsRefracted_shouldComputeRefractionDeflection(self):
        n1, n2 = 1, 1.5
        intersection = self._createIntersection(n1, n2)
        self.fresnelIntersect._getIsReflected = lambda: False

        fresnelIntersection = self.fresnelIntersect.compute(self.rayAt45, intersection)

        expectedDeflection = math.pi / 4 - math.asin(n1 / n2 * math.sin(math.pi / 4))
        self.assertFalse(fresnelIntersection.isReflected)
        self.assertAlmostEqual(expectedDeflection, fresnelIntersection.angleDeflection)

    def testGivenAnAngleOfIncidenceAboveTotalInternalReflection_shouldHaveAReflectionCoefficientOf1(self):
        n1, n2 = math.sqrt(2), 1.0
        intersection = self._createIntersection(n1, n2)

        self.fresnelIntersect.compute(self.rayAt45, intersection)

        self.assertEqual(1, self.fresnelIntersect._getReflectionCoefficient())

    def testGivenSameRefractiveIndices_shouldHaveAReflectionCoefficientOf0(self):
        n1, n2 = 1.4, 1.4
        intersection = self._createIntersection(n1, n2)

        self.fresnelIntersect.compute(self.rayAt45, intersection)

        self.assertEqual(0, self.fresnelIntersect._getReflectionCoefficient())

    def testGivenPerpendicularIncidence_shouldHaveReflectionCoefficient(self):
        n1, n2 = 1.0, 1.5
        intersection = self._createIntersection(n1, n2)
        rayPerpendicular = Vector(0, 0, -1)

        self.fresnelIntersect.compute(rayPerpendicular, intersection)

        R = (n2-n1)/(n2+n1)
        self.assertEqual(R ** 2, self.fresnelIntersect._getReflectionCoefficient())

    def testIfGoingInside_shouldSetNextMaterialAsMaterialInsideSurface(self):
        n1, n2 = 1.0, 1.5
        intersection = self._createIntersection(n1, n2)

        fresnelIntersection = self.fresnelIntersect.compute(self.rayAt45, intersection)

        self.assertEqual(n2, fresnelIntersection.nextEnvironment.material.index)

    def testIfGoingOutside_shouldSetNextMaterialAsMaterialOutsideSurface(self):
        n1, n2 = 1.0, 1.5
        surfaceNormalAlongRayDirection = Vector(0, 0, -1)
        intersection = self._createIntersection(n1, n2, normal=surfaceNormalAlongRayDirection)

        fresnelIntersection = self.fresnelIntersect.compute(self.rayAt45, intersection)

        self.assertEqual(n1, fresnelIntersection.nextEnvironment.material.index)

    @staticmethod
    def _createIntersection(n1=1.0, n2=1.5, normal=Vector(0, 0, 1)):
        insideEnvironment = Environment(ScatteringMaterial(index=n2))
        outsideEnvironment = Environment(ScatteringMaterial(index=n1))
        surfaceElement = Polygon([Vector()], normal, insideEnvironment, outsideEnvironment)
        return Intersection(10, Vector(0, 0, 0), surfaceElement)
