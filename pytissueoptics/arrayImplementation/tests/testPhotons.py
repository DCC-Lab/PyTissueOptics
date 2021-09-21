import unittest
from arrayImplementation.photons import Photons
from arrayImplementation.vectors import Vectors
from arrayImplementation.scalars import Scalars
import numpy as np


class TestPhotons(unittest.TestCase):

    def setUp(self):
        self.photons = Photons()

    def tearDown(self):
        del self.photons

    def testRouletteLogic(self):
        """For test purposes, we cannot test with random values. The logic is the same than in the Photon class."""
        self.photons.weight[0:499] = 1e-6
        weightTooLowMask = self.photons.weight <= 1e-4
        photonsKillMask = Scalars([False]*self.photons.N) == True
        photonsKillMask[250:750] = True
        photonsKillMask = weightTooLowMask.logical_and(photonsKillMask)
        self.photons.removePhotonsWeights(photonsKillMask)

        self.assertTrue(np.all(np.equal(np.where(self.photons.weight.v == 0, 0, 1)[250:499], 0)))
        self.assertTrue(np.all(np.equal(np.where(self.photons.weight.v == 0, 0, 1)[0:249], 1)))
        self.assertTrue(np.all(np.equal(np.where(self.photons.weight.v == 0, 0, 1)[500:1000], 1)))