import unittest
import math

from pytissueoptics.scene.solids import ThickLens, SymmetricLens, PlanoConvexLens, PlanoConcaveLens
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.material import RefractiveMaterial


class TestThickLens(unittest.TestCase):
    def testGivenAFlatLens_shouldHaveInfiniteFocalLength(self):
        lens = ThickLens(frontRadius=0, backRadius=0, diameter=1, thickness=1,
                         material=RefractiveMaterial(1.5))
        self.assertEqual(math.inf, lens.focalLength)

    def testGivenAFlatLens_shouldHaveCenterThicknessEqualToEdgeThickness(self):
        t = 0.8
        lens = ThickLens(frontRadius=0, backRadius=0, diameter=1, thickness=t)
        self.assertEqual(t, lens.centerThickness)
        self.assertEqual(t, lens.edgeThickness)

    def testGivenALensWithoutRefractiveMaterial_shouldHaveNoFocalLength(self):
        lens = ThickLens(frontRadius=0, backRadius=0, diameter=1, thickness=1)
        with self.assertRaises(ValueError):
            _ = lens.focalLength

    def testGivenBiConvexLens_shouldHaveCorrectFocalLength(self):
        R1, R2, t, n = 10, -8, 0.8, 1.5
        lens = ThickLens(frontRadius=R1, backRadius=R2, diameter=2, thickness=t, material=RefractiveMaterial(n))

        f = 1 / ((n - 1) * (1 / R1 - 1 / R2 + (n - 1) * t / (n * R1 * R2)))
        self.assertAlmostEqual(f, lens.focalLength)

    def testGivenBiConcaveLens_shouldHaveCorrectFocalLength(self):
        R1, R2, t, n = -10, 8, 0.8, 1.5
        lens = ThickLens(frontRadius=R1, backRadius=R2, diameter=2, thickness=t, material=RefractiveMaterial(n))

        f = 1 / ((n - 1) * (1 / R1 - 1 / R2 + (n - 1) * t / (n * R1 * R2)))
        self.assertAlmostEqual(f, lens.focalLength)

    def testGivenPlanoConvexLens_shouldHaveCorrectFocalLength(self):
        R1, R2, t, n = 0, -8, 0.8, 1.5
        lens = ThickLens(frontRadius=R1, backRadius=R2, diameter=2, thickness=t, material=RefractiveMaterial(n))

        f = R2 / (n - 1)
        self.assertAlmostEqual(f, lens.focalLength)

    def testGivenPlanoConcaveLens_shouldHaveCorrectFocalLength(self):
        R1, R2, t, n = 0, 8, 0.8, 1.5
        lens = ThickLens(frontRadius=R1, backRadius=R2, diameter=2, thickness=t, material=RefractiveMaterial(n))

        f = R2 / (n - 1)
        self.assertAlmostEqual(f, lens.focalLength)

    def testGivenALensNotThickEnoughForItsCurvature_shouldRaiseError(self):
        with self.assertRaises(ValueError):
            ThickLens(frontRadius=3, backRadius=-3, diameter=2, thickness=0.1)

    def testShouldHaveCenterThicknessEqualToDesiredThickness(self):
        t = 0.8
        lens = ThickLens(frontRadius=3, backRadius=-3, diameter=1, thickness=t)
        self.assertAlmostEqual(t, lens.centerThickness)

    def testShouldHaveDifferentEdgeThickness(self):
        t = 0.8
        lens = ThickLens(frontRadius=3, backRadius=-3, diameter=1, thickness=t)
        self.assertNotAlmostEqual(t, lens.edgeThickness)

    def testGivenALensWithSurfaceRadiusEqualToLensRadius_shouldRaiseError(self):
        R = 3
        with self.assertRaises(ValueError):
            ThickLens(frontRadius=R, backRadius=-R, diameter=R*2, thickness=R*2)

    def testShouldOnlySmoothFrontAndBackSurfaces(self):
        lens = ThickLens(frontRadius=3, backRadius=-3, diameter=1, thickness=1)
        self.assertTrue(lens.getPolygons("front")[0].toSmooth)
        self.assertTrue(lens.getPolygons("back")[0].toSmooth)
        self.assertFalse(lens.getPolygons("lateral")[0].toSmooth)

    def testShouldHaveFrontSurfaceVerticesLyingOnASphereOfFrontRadiusWithNormalsPointingOutward(self):
        R1 = 3.0
        t = 1
        lens = ThickLens(frontRadius=R1, backRadius=-R1, diameter=R1, thickness=t)
        frontSphereOrigin = lens.position - Vector(0, 0, t / 2 - R1)
        frontPolygons = lens.getPolygons("front")
        frontDirection = lens.direction * -1
        for polygon in frontPolygons:
            self.assertTrue(polygon.normal.dot(frontDirection) > 0)
            for vertex in polygon.vertices:
                difference = vertex - frontSphereOrigin
                self.assertAlmostEqual(R1, difference.getNorm())

    def testShouldHaveBackSurfaceVerticesLyingOnASphereOfBackRadiusWithNormalsPointingOutward(self):
        R1 = 3.0
        t = 1
        lens = ThickLens(frontRadius=R1, backRadius=-R1, diameter=R1, thickness=t)
        backSphereOrigin = lens.position + Vector(0, 0, t / 2 - R1)
        backPolygons = lens.getPolygons("back")
        backDirection = lens.direction
        for polygon in backPolygons:
            self.assertTrue(polygon.normal.dot(backDirection) > 0)
            for vertex in polygon.vertices:
                difference = vertex - backSphereOrigin
                self.assertAlmostEqual(R1, difference.getNorm())

    def testShouldHaveLateralSurfaceVerticesLyingOnACylinderOfRadiusEqualToLensRadiusWithNormalsPointingOutward(self):
        r = 1
        t = 1
        lens = ThickLens(frontRadius=3, backRadius=-3, diameter=r*2, thickness=t)
        lateralPolygons = lens.getPolygons("lateral")
        for polygon in lateralPolygons:
            self.assertTrue(polygon.normal.dot(lens.direction) == 0)
            for vertex in polygon.vertices:
                self.assertAlmostEqual(r, vertex.x**2 + vertex.y**2)


class TestSymmetricLens(unittest.TestCase):
    def testGivenPositiveFocalLength_shouldCreateSymmetricThickLensWithDesiredFocalLength(self):
        f, t, n = 10.0, 0.8, 1.5
        lens = SymmetricLens(f=f, diameter=3, thickness=t, material=RefractiveMaterial(n))
        self.assertAlmostEqual(f, lens.focalLength)
        self.assertTrue(lens.frontRadius > 0)
        self.assertAlmostEqual(lens.frontRadius, -lens.backRadius)
        self.assertAlmostEqual(t, lens.centerThickness)

    def testGivenNegativeFocalLength_shouldCreateSymmetricThickLensWithDesiredFocalLength(self):
        f, t, n = -10.0, 0.8, 1.5
        lens = SymmetricLens(f=f, diameter=3, thickness=t, material=RefractiveMaterial(n))
        self.assertAlmostEqual(f, lens.focalLength)
        self.assertTrue(lens.frontRadius < 0)
        self.assertAlmostEqual(lens.frontRadius, -lens.backRadius)
        self.assertAlmostEqual(t, lens.centerThickness)


class TestPlanoConvexLens(unittest.TestCase):
    def testGivenPositiveFocalLength_shouldCreatePlanoConvexLensWithDesiredFocalLength(self):
        f, t, n = 10.0, 0.8, 1.5
        lens = PlanoConvexLens(f=f, diameter=3, thickness=t, material=RefractiveMaterial(n))
        self.assertAlmostEqual(f, lens.focalLength)
        self.assertTrue(lens.frontRadius > 0)
        self.assertEqual(math.inf, lens.backRadius)
        self.assertAlmostEqual(t, lens.centerThickness)

    def testGivenNegativeFocalLength_shouldCreatePlanoConvexLensWithDesiredFocalLength(self):
        f, t, n = -10.0, 0.8, 1.5
        lens = PlanoConvexLens(f=f, diameter=3, thickness=t, material=RefractiveMaterial(n))
        self.assertAlmostEqual(f, lens.focalLength)
        self.assertTrue(lens.backRadius < 0)
        self.assertEqual(math.inf, lens.frontRadius)
        self.assertAlmostEqual(t, lens.centerThickness)


class TestPlanoConcaveLens(unittest.TestCase):
    def testGivenPositiveFocalLength_shouldCreatePlanoConcaveLensWithDesiredFocalLength(self):
        f, t, n = 10.0, 0.8, 1.5
        lens = PlanoConcaveLens(f=f, diameter=3, thickness=t, material=RefractiveMaterial(n))
        self.assertAlmostEqual(f, lens.focalLength)
        self.assertTrue(lens.backRadius > 0)
        self.assertEqual(math.inf, lens.frontRadius)
        self.assertAlmostEqual(t, lens.centerThickness)

    def testGivenNegativeFocalLength_shouldCreatePlanoConcaveLensWithDesiredFocalLength(self):
        f, t, n = -10.0, 0.8, 1.5
        lens = PlanoConcaveLens(f=f, diameter=3, thickness=t, material=RefractiveMaterial(n))
        self.assertAlmostEqual(f, lens.focalLength)
        self.assertTrue(lens.frontRadius < 0)
        self.assertEqual(math.inf, lens.backRadius)
        self.assertAlmostEqual(t, lens.centerThickness)
