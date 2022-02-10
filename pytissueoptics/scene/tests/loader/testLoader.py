import unittest
from pytissueoptics.scene.loader import Loader


class TestLoader(unittest.TestCase):
    def testWhenWrongExtension_shouldRaiseError(self):
        with self.assertRaises(ValueError):
            _ = Loader().load("test.wrongExtension")

    def testWhenLoadingOBJ_shouldNotRaiseError(self):
        loader = Loader()
        _ = loader.load("./parsers/objFiles/droid.obj")

    def testWhenLoadingMultiPolygonObject_shouldSplitInTriangles(self):
        loader = Loader()
        solidObjects = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")
        self.assertEqual(13, len(solidObjects[0].surfaces))

    def testWhenLoadingMultiGroupObject_shouldSplitCorrectGroups(self):
        loader = Loader()
        solidObjects = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")
        self.assertCountEqual(solidObjects[0].groups, ["front", "back", "bottom", "top", "right", "left"])

    def testWhenLoadingMultiGroupObject_shouldHaveCorrectAmountOfElementsPerGroup(self):
        loader = Loader()
        solidObjects = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")
        self.assertEqual(len(solidObjects[0]._surfaceDict["front"]), 2)
        self.assertEqual(len(solidObjects[0]._surfaceDict["back"]), 3)
