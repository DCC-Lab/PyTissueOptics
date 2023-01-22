import os
import unittest
from typing import List

from pytissueoptics.scene.loader import Loader
from pytissueoptics.scene.solids import Solid


class TestLoader(unittest.TestCase):
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    def testWhenLoadWithWrongExtension_shouldNotLoad(self):
        with self.assertRaises(NotImplementedError):
            _ = Loader().load(self._filepath("test.wrongExtension"), showProgress=False)

    def testWhenLoadingOBJ_shouldLoad(self):
        loader = Loader()
        _ = loader.load(self._filepath("testCubeTrianglesMulti.obj"), showProgress=False)

    def testWhenLoadingOBJ_shouldReturnListOfSolids(self):
        loader = Loader()
        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"), showProgress=False)
        self.assertIsInstance(solids, List)
        for solid in solids:
            self.assertIsInstance(solid, Solid)

    def testWhenLoadingMultiPolygonObject_shouldSplitInTriangles(self):
        loader = Loader()
        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"), showProgress=False)
        self.assertEqual(1, len(solids))
        self.assertEqual(13, len(solids[0].getPolygons()))

    def testWhenLoadingMultiGroupObject_shouldSplitCorrectGroups(self):
        loader = Loader()
        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"), showProgress=False)
        self.assertCountEqual(["front", "back", "bottom", "top", "right", "left"], solids[0].surfaceLabels)

    def testWhenLoadingMultiGroupObject_shouldHaveCorrectAmountOfElementsPerGroup(self):
        loader = Loader()

        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"), showProgress=False)

        self.assertEqual(2, len(solids[0].surfaces.getPolygons("front")))
        self.assertEqual(3, len(solids[0].surfaces.getPolygons("back")))

    def _filepath(self, fileName) -> str:
        return os.path.join(self.TEST_DIRECTORY, "parsers", "objFiles", fileName)
