import unittest

from python_graphics_engine.geometry import Vector
from python_graphics_engine.solids import Cuboid


class TestCuboid(unittest.TestCase):
    def setUp(self):
        self.position = Vector(2, 2, 1)
        self.cuboid = Cuboid(8, 1, 3, position=self.position)

    def testGivenANewDefaultCuboid_shouldBePlacedAtOrigin(self):
        cuboid = Cuboid(8, 1, 3)
        self.assertEqual(Vector(0, 0, 0), cuboid.position)
