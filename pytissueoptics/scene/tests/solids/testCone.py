import unittest

from pytissueoptics.scene.geometry import Vector, Vertex, primitives
from pytissueoptics.scene.solids import Cone


class TestCone(unittest.TestCase):
    def testWhenContainsWithVerticesThatAreAllInsideTheCone_shouldReturnTrue(self):
        r = 1
        h = 3
        midRadius = r * 0.5
        f = 0.9
        cylinder = Cone(radius=r, length=h, u=32, v=2, position=Vector(0, 0, 0))

        vertices = [
            Vertex(f * midRadius, 0, 0),
            Vertex(0, f * midRadius, 0),
            Vertex(-f * midRadius, 0, 0),
            Vertex(0, -f * midRadius, 0),
            Vertex(0, 0, f * h * 0.5),
            Vertex(0, 0, -f * h * 0.5),
        ]

        self.assertTrue(cylinder.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotInsideTheCone_shouldReturnFalse(self):
        r = 1
        h = 3
        midRadius = r * 0.5
        f = 1.1
        cylinder = Cone(radius=r, length=h, u=32, v=2, position=Vector(0, 0, 0))

        vertices = [
            Vertex(f * midRadius, 0, 0),
            Vertex(0, f * midRadius, 0),
            Vertex(-f * midRadius, 0, 0),
            Vertex(0, -f * midRadius, 0),
            Vertex(0, 0, f * h * 0.5),
            Vertex(0, 0, -f * h * 0.5),
        ]

        for vertex in vertices:
            self.assertFalse(cylinder.contains(vertex))

    def testGivenANewWithQuadPrimitive_shouldNotCreateCone(self):
        with self.assertRaises(NotImplementedError):
            Cone(primitive=primitives.QUAD)
