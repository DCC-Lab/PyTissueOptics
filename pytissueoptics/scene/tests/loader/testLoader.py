import unittest
from typing import List
from pytissueoptics.scene.loader import Loader
from pytissueoptics.scene.solids import Solid


class TestLoader(unittest.TestCase):
    def testWhenLoadWithWrongExtension_shouldNotLoad(self):
        with self.assertRaises(ValueError):
            _ = Loader().load("test.wrongExtension")

    def testWhenLoadingOBJ_shouldLoad(self):
        loader = Loader()
        _ = loader.load("./parsers/objFiles/droid.obj")

    def testWhenLoadingOBJ_shouldReturnListOfSolids(self):
        loader = Loader()
        solids = loader.load("./parsers/objFiles/droid.obj")
        self.assertIsInstance(solids, List)
        for solid in solids:
            self.assertIsInstance(solid, Solid)

    def testWhenLoadingMultiPolygonObject_shouldSplitInTriangles(self):
        loader = Loader()
        solids = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")
        self.assertEqual(13, len(solids[0].surfaces))

    def testWhenLoadingMultiGroupObject_shouldSplitCorrectGroups(self):
        loader = Loader()
        solids = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")
        self.assertCountEqual(solids[0].groups, ["front", "back", "bottom", "top", "right", "left"])

    def testWhenLoadingMultiGroupObject_shouldHaveCorrectAmountOfElementsPerGroup(self):
        loader = Loader()

        solidObjects = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")

        self.assertEqual(len(solidObjects[0]._surfaceDict["front"]), 2)
        self.assertEqual(len(solidObjects[0]._surfaceDict["back"]), 3)
