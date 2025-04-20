import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics import Direction
from pytissueoptics.rayscattering.display.profiles import Profile1D
from pytissueoptics.rayscattering.tests import SHOW_VISUAL_TESTS
from pytissueoptics.scene.tests import compareVisuals

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

        if not SHOW_VISUAL_TESTS:
            self.skipTest("Visual tests are disabled. Set rayscattering.tests.SHOW_VISUAL_TESTS to True to enable them.")

        with tempfile.TemporaryDirectory() as tempdir:
            currentImage = os.path.join(tempdir, 'test.png')
            plt.savefig(currentImage)
            plt.close()

            isOK = compareVisuals(referenceImage, currentImage,
                                  title="TestView2D: View2DProjectionX")
        if not isOK:
            self.fail("Visual test failed.")
