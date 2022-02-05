import unittest
from pytissueoptics.scene.loaders import Loader
#from pytissueoptics.scene.viewer.mayavi import May


class TestLoader(unittest.TestCase):
    def testWhenWrongExtension_shouldRaiseError(self):
        with self.assertRaises(ValueError):
            solidObject = Loader().load("test.wrongExtension")

    def testWhenWrongExtension_shouldRaiseError(self):
        loader = Loader()
        solidObject = loader.load("./parsers/testCubeQuads.obj")
