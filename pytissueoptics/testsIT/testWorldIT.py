import unittest

from pytissueoptics import *


class TestWorldIT(unittest.TestCase):
    def testShouldHaveNoPhotons(self):
        world = World()
        self.assertEqual(0, len(world.photons))

    def testGivenNoGeometries_whenCompute_shouldRaiseException(self):
        pass

    def testGivenNoSource_whenCompute_shouldRaiseException(self):
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


if __name__ == '__main__':
    unittest.main()
