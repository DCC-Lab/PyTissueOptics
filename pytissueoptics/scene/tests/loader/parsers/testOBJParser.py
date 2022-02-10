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

    def testWithTestCube_ShouldGiveCorrectVertices(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        vertices = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        self.assertCountEqual(vertices, parser.vertices)

    def testWithTestCube_ShouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(6, len(parser.normals))

    def testWithTestCube_ShouldGiveCorrectNormals(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        normals = [[0, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        self.assertCountEqual(normals, parser.normals)

    def testWithTestCube_shouldGiveCorrectVerticesIndexForPolygonsWithShiftOf1(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        # indicesBackOBJFile = [5, 6, 7, 8]
        indicesBackInParser = [[4, 5, 6, 7]]
        self.assertCountEqual(indicesBackInParser, parser.objects["cube"].surfaces["back"].polygons)
