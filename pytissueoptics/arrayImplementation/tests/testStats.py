import unittest
from arrayImplementation.photons import Photons
from arrayImplementation.vectors import Vectors
from arrayImplementation.scalars import Scalars
import numpy as np


class TestStats(unittest.TestCase):

    def setUp(self):
        self.photons = Photons()

    def tearDown(self):
        del self.photons

