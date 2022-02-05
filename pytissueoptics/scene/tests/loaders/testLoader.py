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
        for obj in solidObjects:
            viewer.addMayaviSolid(MayaviSolid(obj), representation="surface")
        

        viewer.show()
