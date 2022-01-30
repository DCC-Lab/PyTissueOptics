import unittest

from graphics.solids import Cuboid


class TestCuboid(unittest.TestCase):
    def testGivenNewCuboid(self):
        cuboid = Cuboid(8, 1, 3)
