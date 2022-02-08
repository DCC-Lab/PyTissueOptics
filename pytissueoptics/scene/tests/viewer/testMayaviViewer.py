import unittest
from unittest.mock import patch

from pytissueoptics.scene import Cube
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.viewer.mayavi import MayaviViewer


class TestMayaviViewer(unittest.TestCase):
    @patch('mayavi.mlab.triangular_mesh')
    def testWhenAddSolidWithTrianglePrimitive_shouldAddTheSolid(self, fakeMayaviTriangularMesh):
        viewer = MayaviViewer()
        solid = Cube(3, primitive=primitives.TRIANGLE)

        viewer.add(solid)

        self.assertEqual(1, fakeMayaviTriangularMesh.call_count)

    @patch('mayavi.mlab.triangular_mesh')
    def testWhenAddSolidWithoutTrianglePrimitive_shouldNotAddTheSolid(self, fakeMayaviTriangularMesh):
        viewer = MayaviViewer()
        solid = Cube(3, primitive=primitives.QUAD)

        with self.assertRaises(Exception):
            viewer.add(solid)
        self.assertEqual(0, fakeMayaviTriangularMesh.call_count)

    @patch('mayavi.mlab.show')
    def testWhenShow_shouldDisplayTheMayaviViewer(self, fakeShow):
        viewer = MayaviViewer()
        solid = Cube(3, primitive=primitives.TRIANGLE)
        viewer.add(solid)

        viewer.show()

        self.assertEqual(1, fakeShow.call_count)
