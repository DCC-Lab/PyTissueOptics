import unittest
from unittest.mock import patch

from python_graphics_engine.geometry import Vector, Quad, primitives
from python_graphics_engine.solids import Solid


class TestSolid(unittest.TestCase):
    def setUp(self):
        a, b, c = 2, 2, 2
        self.CUBOID_VERTICES = [Vector(-a / 2, -b / 2, -c / 2), Vector(a / 2, -b / 2, -c / 2),
                                Vector(a / 2, b / 2, -c / 2), Vector(-a / 2, b / 2, -c / 2),
                                Vector(-a / 2, -b / 2, c / 2), Vector(a / 2, -b / 2, c / 2),
                                Vector(a / 2, b / 2, c / 2), Vector(-a / 2, b / 2, c / 2)]
        V = self.CUBOID_VERTICES
        self.CUBOID_SURFACES = {'Front': [Quad(V[0], V[1], V[2], V[3])], 'Back': [Quad(V[5], V[4], V[7], V[6])],
                                'Left': [Quad(V[4], V[0], V[3], V[7])], 'Right': [Quad(V[1], V[5], V[6], V[2])],
                                'Top': [Quad(V[3], V[2], V[6], V[7])], 'Bottom': [Quad(V[4], V[5], V[1], V[0])]}

        # maybe create mock instances of vertices and surfaces...
        # for the vertices we are already testing .add and all in testVector
        #  so we just need to test that it tells the vertices to do the proper thing

    @patch('python_graphics_engine.solids.Solid._computeMesh')
    def testGivenASolid(self, fakeComputeMesh):
        solid = Solid(position=Vector(0, 0, 0), material=None, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)
        self.assertEqual(Vector(0, 0, 0), solid.position)
