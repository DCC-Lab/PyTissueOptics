import unittest
from pytissueoptics.scene.loaders import Loader
from pytissueoptics.scene.viewer.mayavi import MayaviSolid, MayaviViewer


class TestLoader(unittest.TestCase):
    def testWhenWrongExtension_shouldRaiseError(self):
        with self.assertRaises(ValueError):
            solidObject = Loader().load("test.wrongExtension")

    def testWhenWrongExtension_shouldRaiseError(self):
        loader = Loader()
        solidObjects = loader.load("./parsers/droid.obj")
        viewer = MayaviViewer()

        viewer.addMayaviSolid(MayaviSolid(solidObjects[0]), representation="surface")
        viewer.addMayaviSolid(MayaviSolid(solidObjects[1]), representation="surface")
        viewer.addMayaviSolid(MayaviSolid(solidObjects[2]), representation="surface")

        viewer.show()
