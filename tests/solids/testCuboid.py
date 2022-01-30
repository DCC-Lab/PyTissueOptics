import unittest

from python_graphics_engine.solids import Cuboid


class TestCuboid(unittest.TestCase):
    def testGivenNewCuboid(self):
        cuboid = Cuboid(8, 1, 3)
