import os
import unittest

from pytissueoptics.scene.loader import loadSolid


class TestLoadSolid(unittest.TestCase):
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    def testWhenLoadingFromOBJ_shouldReturnASolidOfThisFile(self):
        solid = loadSolid(self._filepath("testCubeTrianglesMulti.obj"), showProgress=False)
        self.assertIsNotNone(solid)

        self.assertEqual(13, len(solid.getPolygons()))
        self.assertEqual(
            ["cube_front", "cube_back", "cube_left", "cube_right", "cube_top", "cube_bottom"], solid.surfaceLabels
        )

    def _filepath(self, fileName) -> str:
        return os.path.join(self.TEST_DIRECTORY, "parsers", "objFiles", fileName)
