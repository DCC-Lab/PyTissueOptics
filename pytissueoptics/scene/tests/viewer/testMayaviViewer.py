import unittest
from unittest.mock import patch

from pytissueoptics.scene import Cube
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.viewer.mayavi import MayaviViewer


class TestMayaviViewer(unittest.TestCase):
    @patch('mayavi.mlab.figure')
    def setUp(self, fakeFigure):
        self.viewer = MayaviViewer()

    @patch('mayavi.mlab.triangular_mesh')
    def testWhenAddSolidWithTrianglePrimitive_shouldAddTheSolid(self, fakeMayaviTriangularMesh):
        solid = Cube(3, primitive=primitives.TRIANGLE)

        self.viewer.add(solid)

        self.assertEqual(1, fakeMayaviTriangularMesh.call_count)

    @patch('mayavi.mlab.triangular_mesh')
    def testWhenAddSolidWithoutTrianglePrimitive_shouldNotAddTheSolid(self, fakeMayaviTriangularMesh):
        solid = Cube(3, primitive=primitives.QUAD)

        with self.assertRaises(Exception):
            self.viewer.add(solid)
        self.assertEqual(0, fakeMayaviTriangularMesh.call_count)

    @patch('mayavi.mlab.show')
    def testWhenShow_shouldDisplayTheMayaviViewer(self, fakeShow):
        solid = Cube(3, primitive=primitives.TRIANGLE)
        self.viewer.add(solid)

        self.viewer.show()

        self.assertEqual(1, fakeShow.call_count)
