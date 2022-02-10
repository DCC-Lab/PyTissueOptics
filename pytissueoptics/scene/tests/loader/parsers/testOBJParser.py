import unittest
from pytissueoptics.scene.loader.parsers import OBJParser


class TestOBJParser(unittest.TestCase):
    def testGivenNoFilepath_shouldRaiseException(self):
        with self.assertRaises(Exception):
            parser = OBJParser()

    def testGivenWrongExtension_shouldRaiseTypeError(self):
        with self.assertRaises(TypeError):
            parser = OBJParser("objFiles/test.wrongExtension")

    def testAtCreation_ShouldFillAttributes(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertTrue(parser.objects)
        self.assertTrue(parser.vertices)
        self.assertTrue(parser.normals)

    def testWithTestCube_ShouldGiveCorrectAmountOfVertices(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(8, len(parser.vertices))

    def testWithTestCube_ShouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(6, len(parser.normals))

