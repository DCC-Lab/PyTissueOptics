import unittest
from pytissueoptics.scene.loaders import OBJParser


class TestOBJParser(unittest.TestCase):
    def testWhenCreatedEmpty_shouldRaiseError(self):
        with self.assertRaises(Exception):
            parser = OBJParser()


    def testWhenWrongExtension_shouldRaiseError(self):
        with self.assertRaises(TypeError):
            parser = OBJParser("nightstand.dae")

    def testWhenCorrectExtension_shouldNotDoAnything(self):
        parser = OBJParser("nightstand.obj")
