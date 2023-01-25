import filecmp
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics import Direction
from pytissueoptics.rayscattering.display.profiles import Profile1D

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'testImages')

OVERWRITE_REFERENCE_IMAGES = False


class TestProfile1D(unittest.TestCase):
    def testWhenShow_shouldPlotTheProfileWithCorrectDataAndAxis(self):
        profile = Profile1D(np.array([1, 2, 8, 4, 1]), Direction.X_NEG,
                            limits=(0, 1), name='Test Profile along X_NEG')

        with patch("matplotlib.pyplot.show") as mockShow:
            profile.show()
            mockShow.assert_called_once()

        referenceImage = os.path.join(TEST_IMAGES_DIR, 'profile1D.png')

        if OVERWRITE_REFERENCE_IMAGES:
            plt.savefig(referenceImage)
            self.skipTest("Overwriting reference image")

        with tempfile.TemporaryDirectory() as tempdir:
            currentImage = os.path.join(tempdir, 'profile1D.png')
            plt.savefig(currentImage)
            self.assertTrue(filecmp.cmp(referenceImage, currentImage))
        plt.close()
