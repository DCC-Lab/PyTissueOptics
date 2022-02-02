import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid


class TestCuboid(unittest.TestCase):
    def testGivenANewDefaultCuboid_shouldBePlacedAtOrigin(self):
        cuboid = Cuboid(8, 1, 3)
        self.assertEqual(Vector(0, 0, 0), cuboid.position)

    def testGivenANewCuboid_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        cuboid = Cuboid(8, 1, 3, position=position)
        self.assertEqual(position, cuboid.position)
