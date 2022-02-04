import unittest

from pytissueoptics.scene.loaders import OBJParser


class TestOBJParser(unittest.TestCase):
    def testWhenCreated_shouldDefineItsNormal(self):
        triangle = Triangle(v1=Vector(0, 0, 0), v2=Vector(2, 0, 0), v3=Vector(2, 2, 0))
        self.assertEqual(Vector(0, 0, 1), triangle.normal)

