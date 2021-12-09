from mockito import when, mock
import unittest

from pytissueoptics import World, Geometry, Source, Vector, Photon


class TestWorld(unittest.TestCase):
    def testShouldHaveNoPhotons(self):
        world = World()
        self.assertEqual(0, len(world.photons))

    def testGivenNoGeometries_whenRun_shouldRaiseException(self):
        pass

    def testGivenNoSource_whenRun_shouldRaiseException(self):
        pass

    def testGivenGeometryAndSource_whenRun_shouldPropagatePhotons(self):
        world = World()
        source = mock(Source)
        photon1 = Photon()
        photon2 = Photon()
        when(source).getPhotons().thenReturn([photon1, photon2])

        world.place(mock(Geometry), Vector(0, 0, 0))
        world.place(source, Vector(0, 0, 0))
        world.run()


if __name__ == '__main__':
    unittest.main()
