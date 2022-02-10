import os
import unittest

from pytissueoptics.scene.loader.parsers import OBJParser


class TestOBJParser(unittest.TestCase):
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    def testGivenNoFilepath_shouldRaiseException(self):
        with self.assertRaises(Exception):
            parser = OBJParser()

    def testGivenWrongExtension_shouldRaiseTypeError(self):
        with self.assertRaises(TypeError):
            parser = OBJParser(self._filepath("test.wrongExtension"))

    def testAtCreation_shouldFillAttributes(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"))
        self.assertTrue(parser.objects)
        self.assertTrue(parser.vertices)
        self.assertTrue(parser.normals)

    def testWithTestCube_shouldGiveCorrectAmountOfVertices(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"))
        self.assertEqual(8, len(parser.vertices))

    def testWithTestCube_shouldGiveCorrectVertices(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"))
        vertices = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        self.assertCountEqual(vertices, parser.vertices)

    def testWithTestCube_shouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"))
        self.assertEqual(6, len(parser.normals))

    def testWithTestCube_shouldGiveCorrectNormals(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"))
        normals = [[0, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        self.assertCountEqual(normals, parser.normals)

    def testWithTestCube_shouldGiveCorrectVerticesIndexForPolygonsWithShiftOf1(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"))
        indicesBackInParser = [[4, 5, 6, 7]]
        self.assertCountEqual(indicesBackInParser, parser.objects["cube"].surfaces["back"].polygons)

    def _filepath(self, fileName) -> str:
        return os.path.join(self.TEST_DIRECTORY, "objFiles", fileName)
