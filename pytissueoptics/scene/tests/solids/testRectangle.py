import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Rectangle


class TestRectangle(unittest.TestCase):
    def testShouldBeFlat(self):
        rectangle = Rectangle(2, 2)
        self.assertTrue(rectangle.isFlat)

    def testShouldHaveDefaultZOrientation(self):
        rectangle = Rectangle(2, 2)
        polygons = rectangle.getPolygons()
        normals = [p.normal for p in polygons]
        for n in normals:
            self.assertEqual(n, Vector(0, 0, 1))

    def testGivenAnOrientation_shouldOrientTheRectangle(self):
        orientation = Vector(1, 1, 0)
        orientation.normalize()
        rectangle = Rectangle(2, 2, orientation=orientation)
        polygons = rectangle.getPolygons()
        normals = [p.normal for p in polygons]
        for n in normals:
            self.assertEqual(n, orientation)

    def testShouldHaveASingleSurface(self):
        rectangle = Rectangle(2, 2)
        assert len(rectangle.surfaceLabels) == 1

    def testShouldHaveDifferentHashWhenOrientationChanges(self):
        rectangle1 = Rectangle(2, 2, orientation=Vector(0, 0, 1))
        rectangle2 = Rectangle(2, 2, orientation=Vector(1, 1, 0))
        self.assertNotEqual(hash(rectangle1), hash(rectangle2))
        rectangle3 = Rectangle(2, 2, orientation=Vector(0, 0, 1))
        self.assertEqual(hash(rectangle1), hash(rectangle3))
