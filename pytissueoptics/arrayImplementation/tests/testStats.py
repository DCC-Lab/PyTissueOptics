import unittest
from arrayImplementation.photons import Photons


class TestStats(unittest.TestCase):

    def setUp(self):
        self.photons = Photons()

    def tearDown(self):
        del self.photons

