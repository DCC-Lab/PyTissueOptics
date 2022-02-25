import os
import unittest
from typing import List

from pytissueoptics.scene.loader import Loader
from pytissueoptics.scene.solids import Solid


class TestLoader(unittest.TestCase):
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    def testWhenLoadWithWrongExtension_shouldNotLoad(self):
        with self.assertRaises(ValueError):
            _ = Loader().load(self._filepath("test.wrongExtension"))

    def testWhenLoadingOBJ_shouldLoad(self):
        loader = Loader()
        _ = loader.load(self._filepath("droid.obj"))

    def testWhenLoadingOBJ_shouldReturnListOfSolids(self):
        loader = Loader()
        solids = loader.load(self._filepath("droid.obj"))
        self.assertIsInstance(solids, List)
        for solid in solids:
            self.assertIsInstance(solid, Solid)

    def testWhenLoadingMultiPolygonObject_shouldNotSplitInTriangles(self):
        loader = Loader()
        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"))
        self.assertEqual(11, len(solids[0].getPolygons()))

    def testWhenLoadingMultiGroupObject_shouldSplitCorrectGroups(self):
        loader = Loader()
        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"))
        self.assertCountEqual(["front", "back", "bottom", "top", "right", "left"], solids[0].surfaceNames)

    def testWhenLoadingMultiGroupObject_shouldHaveCorrectAmountOfElementsPerGroup(self):
        loader = Loader()

        solids = loader.load(self._filepath("testCubeTrianglesMulti.obj"))

        self.assertEqual(2, len(solids[0].surfaces.getPolygons("front")))
        self.assertEqual(1, len(solids[0].surfaces.getPolygons("back")))

    def _filepath(self, fileName) -> str:
        return os.path.join(self.TEST_DIRECTORY, "parsers", "objFiles", fileName)
