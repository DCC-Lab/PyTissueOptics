import math
import unittest

from pytissueoptics.scene import Vector
from pytissueoptics.scene.intersection import Ray


class TestRay(unittest.TestCase):
    def testGivenNewRay_shouldNormalizeItsDirection(self):
        ray = Ray(origin=Vector(0, 0, 0), direction=Vector(2, 2, 0), length=10)
        self.assertEqual(Vector(math.sqrt(2) / 2, math.sqrt(2) / 2, 0), ray.direction)
