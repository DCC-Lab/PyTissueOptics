import unittest
from pytissueoptics.scene.loader.parsers import OBJParser


class TestOBJParser(unittest.TestCase):
    def testGivenNoFilepath_shouldNotParse(self):
        with self.assertRaises(Exception):
            parser = OBJParser()

    def testGivenWrongExtension_shouldNotParse(self):
        with self.assertRaises(TypeError):
            parser = OBJParser("objFiles/test.wrongExtension")

    def testShouldParseAtCreation(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")

    def testShouldGiveCorrectAmountOfVertices(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(8, len(parser.vertices))

    def testShouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(6, len(parser.normals))
