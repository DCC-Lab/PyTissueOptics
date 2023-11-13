import unittest
import math

from pytissueoptics.scene.geometry import Vector, Vertex, primitives
from pytissueoptics.scene.solids import Cylinder


class TestCylinder(unittest.TestCase):
    def testGivenANewDefault_shouldBePlacedAtOrigin(self):
        cylinder = Cylinder()
        self.assertEqual(Vector(0, 0, 0), cylinder.position)
        self.assertEqual(Vector(0, 0, 0), cylinder.bbox.center)

    def testGivenANew_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        cylinder = Cylinder(position=position)
        self.assertEqual(Vector(2, 2, 1), cylinder.position)

    def testGivenANewWithQuadPrimitive_shouldNotCreateCylinder(self):
        with self.assertRaises(NotImplementedError):
            Cylinder(primitive=primitives.QUAD)

    def testGivenANewWithUSmallerThan3_shouldNotCreateCylinder(self):
        with self.assertRaises(ValueError):
            Cylinder(u=2)

    def testGivenANewWithVSmallerThan1_shouldNotCreateCylinder(self):
        with self.assertRaises(ValueError):
            Cylinder(v=0)

    def testGivenALowOrderCylinder_shouldApproachCorrectCylinderAreaTo5Percent(self):
        cylinder = Cylinder(radius=1, length=2, u=12)
        perfectCylinderArea = (2 * math.pi) + (2 * math.pi * 2)
        tolerance = 0.05
        cylinderArea = self._getTotalTrianglesArea(cylinder.getPolygons())

        self.assertAlmostEqual(perfectCylinderArea, cylinderArea, delta=tolerance * perfectCylinderArea)

    def testGivenAHighOrderCylinder_shouldApproachCorrectCylinderAreaTo1TenthOfAPercent(self):
        cylinder = Cylinder(radius=1, length=2, u=100)
        perfectCylinderArea = (2 * math.pi) + (2 * math.pi * 2)
        tolerance = 0.001
        cylinderArea = self._getTotalTrianglesArea(cylinder.getPolygons())

        self.assertAlmostEqual(perfectCylinderArea, cylinderArea, delta=tolerance * perfectCylinderArea)

    @staticmethod
    def _getTotalTrianglesArea(surfaces):
        totalArea = 0
        for surface in surfaces:
            AB = surface.vertices[0] - surface.vertices[1]
            AC = surface.vertices[0] - surface.vertices[2]
            totalArea += 0.5 * AB.cross(AC).getNorm()
        return totalArea

    def testWhenContainsWithVerticesThatAreAllInsideTheCylinder_shouldReturnTrue(self):
        cylinder = Cylinder(radius=1, length=3, u=32, v=2, position=Vector(2, 2, 0))
        cylinder.rotate(0, 45, 0)
        vertices = [Vertex(2+1, 2, 1), Vertex(2, 2, 0.5)]

        self.assertTrue(cylinder.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotAllInsideTheCylinder_shouldReturnFalse(self):
        cylinder = Cylinder(radius=1, length=3, u=32, v=2, position=Vector(2, 2, 0))
        cylinder.rotate(0, 30, 0)
        vertices = [Vertex(3.51, 2, 2.6), Vertex(2, 2, 0.5)]
                
        self.assertFalse(cylinder.contains(*vertices))

    def testWhenContainsWithVerticesOutsideMinRadius_shouldReturnFalse(self):
        r = 1000
        h = 3
        minRadiusWith6Divisions = 0.866
        f = minRadiusWith6Divisions * 1.01
        cylinder = Cylinder(radius=r, length=h, u=6, position=Vector(0, 0, 0))
        
        vertices = [Vertex(f * r, 0, 0), Vertex(0, f * r, 0), Vertex(0, 0, h * 0.51),
                    Vertex(-f * r, 0, 0), Vertex(0, -f * r, 0), Vertex(0, 0, -h * 0.51)]
        
        for vertex in vertices:
            self.assertFalse(cylinder.contains(vertex))

    def testWhenContainsWithVerticesInsideMinRadius_shouldReturnTrue(self):
        r = 1000
        h = 3
        minRadiusWith6Divisions = 0.866
        f = minRadiusWith6Divisions * 0.99
        cylinder = Cylinder(radius=r, length=h, u=6, position=Vector(0, 0, 0))
        
        vertices = [Vertex(f * r, 0, 0), Vertex(0, f * r, 0), Vertex(0, 0, h * 0.49),
                    Vertex(-f * r, 0, 0), Vertex(0, -f * r, 0), Vertex(0, 0, -h * 0.49)]

        self.assertTrue(cylinder.contains(*vertices))

    def testWhenSmoothWithLessThan16Sides_shouldWarn(self):
        with self.assertWarns(UserWarning):
            Cylinder(u=15, smooth=True)
