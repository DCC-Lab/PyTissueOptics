import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Circle


class TestCircle(unittest.TestCase):
    def testShouldBeFlat(self):
        circle = Circle(radius=1)
        self.assertTrue(circle.isFlat)

    def testShouldHaveDefaultZOrientation(self):
        circle = Circle(radius=1)
        polygons = circle.getPolygons()
        normals = [p.normal for p in polygons]
        for n in normals:
            self.assertEqual(n, Vector(0, 0, 1))

    def testGivenAnOrientation_shouldOrientTheCircle(self):
        orientation = Vector(1, 1, 0)
        orientation.normalize()
        circle = Circle(radius=1, orientation=orientation)
        polygons = circle.getPolygons()
        normals = [p.normal for p in polygons]
        for n in normals:
            self.assertEqual(n, orientation)

    def testShouldHaveASingleSurface(self):
        circle = Circle(radius=1)
        assert len(circle.surfaceLabels) == 1

    def testShouldHaveDifferentHashWhenOrientationChanges(self):
        circle1 = Circle(radius=1, orientation=Vector(0, 0, 1))
        circle2 = Circle(radius=1, orientation=Vector(1, 1, 0))
        self.assertNotEqual(hash(circle1), hash(circle2))
        circle3 = Circle(radius=1, orientation=Vector(0, 0, 1))
        self.assertEqual(hash(circle1), hash(circle3))
