import os
import unittest

from pytissueoptics.scene.loader.parsers import OBJParser


class TestOBJParser(unittest.TestCase):
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    def testGivenNoFilepath_shouldRaiseException(self):
        with self.assertRaises(Exception):
            _ = OBJParser()

    def testGivenWrongExtension_shouldRaiseTypeError(self):
        with self.assertRaises(TypeError):
            _ = OBJParser(self._filepath("test.wrongExtension"), showProgress=False)

    def testAtCreation_shouldFillAttributes(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        self.assertTrue(parser.objects)
        self.assertTrue(parser.vertices)
        self.assertTrue(parser.normals)

    def testWithTestCube_shouldGiveCorrectAmountOfVertices(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        self.assertEqual(8, len(parser.vertices))

    def testWithTestCube_shouldGiveCorrectVertices(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        vertices = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        self.assertCountEqual(vertices, parser.vertices)

    def testWithTestCube_shouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        self.assertEqual(6, len(parser.normals))

    def testWithTestCube_shouldGiveCorrectNormals(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        normals = [[0, 0, -1], [1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        self.assertCountEqual(normals, parser.normals)

    def testWithTestCube_shouldGiveCorrectVerticesIndexForPolygonsWithShiftOf1(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        indicesBackInParser = [[4, 5, 6, 7]]
        self.assertCountEqual(indicesBackInParser, parser.objects["cube"].surfaces["back"].polygons)

    def testWithTextureCoords_shouldGiveCorrectTextureCoords(self):
        parser = OBJParser(self._filepath("testCubeQuadsTexture.obj"), showProgress=False)
        textureCoords = [[1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]]
        self.assertCountEqual(textureCoords, parser.textureCoords)

    def testWithTestCube_shouldGiveCorrectObjectName(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        self.assertIn("cube", parser.objects)

    def testWithTestCube_shouldGiveCorrectSurfaceNames(self):
        parser = OBJParser(self._filepath("testCubeQuads.obj"), showProgress=False)
        objSurfaces = parser.objects["cube"].surfaces

        for expectedSurface in ["front", "back", "left", "right", "top", "bottom"]:
            self.assertIn(expectedSurface, objSurfaces)

    def testWithNoObjectName_shouldGiveDefaultObjectName(self):
        parser = OBJParser(self._filepath("testCubeQuadsNoObjectName.obj"), showProgress=False)
        self.assertIn(OBJParser.NO_OBJECT, parser.objects)

    def testWithNoSurfaceName_shouldGiveDefaultSurfaceName(self):
        parser = OBJParser(self._filepath("testCubeQuadsNoSurfaceName.obj"), showProgress=False)
        self.assertIn(OBJParser.NO_SURFACE, parser.objects["cube"].surfaces)

    def testWithNoObjectGroup_shouldGiveDefaultObjectGroup(self):
        parser = OBJParser(self._filepath("testCubeQuadsNoObject.obj"), showProgress=False)
        self.assertIn(OBJParser.NO_OBJECT, parser.objects)

    def testWithNoSurfaceGroup_shouldGiveDefaultSurfaceGroup(self):
        parser = OBJParser(self._filepath("testCubeQuadsNoSurface.obj"), showProgress=False)
        self.assertIn(OBJParser.NO_SURFACE, parser.objects["cube"].surfaces)

    def testWithRepeatingSurfaceNames_shouldRenameToUniqueSurfaces(self):
        parser = OBJParser(self._filepath("testCubeQuadsRepeatingSurface.obj"), showProgress=False)
        objSurfaces = parser.objects["cube"].surfaces

        for expectedSurface in ["face", "face_2", "face_3", "face_4", "face_5", "face_6"]:
            self.assertIn(expectedSurface, objSurfaces)

    def _filepath(self, fileName) -> str:
        return os.path.join(self.TEST_DIRECTORY, "objFiles", fileName)
