import unittest

from pytissueoptics import IsotropicSource


class TestIsotropicSourceIT(unittest.TestCase):
    N_PHOTONS = 10

    def setUp(self):
        self.source = IsotropicSource(maxCount=self.N_PHOTONS)

    def testShouldHavePhotons(self):
        photons = self.source.getPhotons()
        self.assertEqual(self.N_PHOTONS, len(photons))

    def testShouldHavePhotonsAtSourceOrigin(self):
        photons = self.source.getPhotons()
        for photon in photons:
            self.assertEqual(self.source.origin, photon.r)

    def testWhenMoveOnePhoton_shouldNotMoveOtherPhotons(self):
        photons = self.source.getPhotons()

        photons[0].moveBy(1)

        for photon in photons[1:]:
            self.assertEqual(self.source.origin, photon.r)
