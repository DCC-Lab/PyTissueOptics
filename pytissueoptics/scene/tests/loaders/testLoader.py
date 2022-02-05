import unittest
from pytissueoptics.scene.loaders import Loader


class TestLoader(unittest.TestCase):
    def testWhenWrongExtension_shouldRaiseError(self):
        with self.assertRaises(ValueError):
            solidObject = Loader().load("test.wrongExtension")

    def testWhenWrongExtension_shouldRaiseError(self):
        solidObject = Loader().load("./parsers/testCubeQuads.obj")

    def testWhenTestCubeQuads_shouldGiveCorrectAmountOfVertices(self):
        parser = OBJParser("testCubeQuads.obj")
        self.assertEqual(8, len(parser.vertices))

    def testWhenTestCubeQuads_shouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser("testCubeQuads.obj")
        self.assertEqual(6, len(parser.normals))
