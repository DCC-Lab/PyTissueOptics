from mockito import when, mock, verify
import unittest

from pytissueoptics import World, Geometry, Source, Vector, Photon, IsotropicSource


class TestWorldIT(unittest.TestCase):
    def testShouldHaveNoPhotons(self):
        world = World()
        self.assertEqual(0, len(world.photons))

    def testGivenNoGeometries_whenRun_shouldRaiseException(self):
        pass

    def testGivenNoSource_whenRun_shouldRaiseException(self):
        pass

    def testGivenSource_shouldHavePhotons(self):
        N = 10
        world = World()
        source = IsotropicSource(maxCount=N)
        world.place(source, Vector(0, 0, 0))

        photons = world.photons

        self.assertEqual(N, len(photons))

    def testGivenSource_shouldInitializePhotonsAtSourceOrigin(self):
        N = 10
        sourceOrigin = Vector(0, 0, 0)
        world = World()
        source = IsotropicSource(maxCount=N)
        world.place(source, sourceOrigin)

        photons = world.photons

        for photon in photons:
            self.assertEqual(sourceOrigin, photon.globalPosition)

    @unittest.skip("")
    def testGivenGeometryAndSource_whenRun_shouldPropagatePhotons(self):
        world = World()
        source = mock(Source)
        photon1 = Photon()
        photon2 = Photon()
        when(source).getPhotons().thenReturn([photon1, photon2])

        world.place(mock(Geometry), Vector(0, 0, 0))
        world.place(source, Vector(0, 0, 0))
        world.simpleCompute()

        for photon in [photon1, photon2]:
            verify(photon).propagate()


if __name__ == '__main__':
    unittest.main()
