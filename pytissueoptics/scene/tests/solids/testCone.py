import unittest

from pytissueoptics.scene.geometry import Vector, Vertex, primitives
from pytissueoptics.scene.solids import Cone


class TestCone(unittest.TestCase):
    def testWhenContainsWithVerticesThatAreAllInsideTheCone_shouldReturnTrue(self):
        cylinder = Cone(radius=1, height=3, u=32, v=2, position=Vector(0, 0, 0))
        vertices = [Vertex(0, 0, 1), Vertex(0, 0.49, 1.5)]
        result = cylinder.contains(*vertices)
        self.assertTrue(result)

    def testWhenContainsWithVerticesThatAreNotAllInsideTheCone_shouldReturnFalse(self):
        cylinder = Cone(radius=1, height=3, u=32, v=2, position=Vector(0, 0, 0))
        vertices = [Vertex(0, 0, 1), Vertex(0, 0.50, 1.5)]
        result = cylinder.contains(*vertices)
        self.assertFalse(result)

    def testGivenANewWithQuadPrimitive_shouldNotCreateCone(self):
        with self.assertRaises(NotImplementedError):
            Cone(primitive=primitives.QUAD)
