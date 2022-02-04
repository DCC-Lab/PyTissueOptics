import unittest


class TestMayaviViewer(unittest.TestCase):
    # todo(?): maybe change interface
    # - move add to private methods
    # - define showScene(), showSolid()
    def testWhenAddMayaviSolidWithTrianglePrimitive_shouldAddTheSolid(self):
        # assert call on mlab.triangular_mesh
        pass

    def testWhenAddMayaviSolidWithoutTrianglePrimitive_shouldNotAddTheSolid(self):
        # assert raises
        pass

    def testWhenShow_shouldDisplayTheMayaviViewer(self):
        # assert call to mlab.show
        pass
