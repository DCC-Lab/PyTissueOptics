import os
import tempfile
import unittest

import numpy as np

from pytissueoptics import (
    Cuboid,
    EnergyLogger,
    PencilPointSource,
    ScatteringMaterial,
    ScatteringScene,
    Sphere,
    Vector,
)
from pytissueoptics.rayscattering.opencl import OPENCL_OK
import pytissueoptics.rayscattering.opencl.CLPhotons as cl_photons
from pytissueoptics.rayscattering.opencl.config.IPPTable import IPPTable
from pytissueoptics.rayscattering.statistics import Stats


@unittest.skipIf(not OPENCL_OK, "OpenCL device not available.")
class TestCLPropagateBVH(unittest.TestCase):
    """
    End-to-end regression: an actual propagate() run via the BVH path must produce statistics
    that match the flat-list path within Monte Carlo noise. This guards against any logic
    divergence between the two intersection backends that the kernel-level parity test might
    miss (e.g. wrong bbox prune, wrong leaf polygon range, stack overflow on real depth).
    """

    def setUp(self):
        # Sibling test (testIPPTable.py) clobbers IPPTable.TABLE_PATH via a temp dir that is
        # already gone by the time we run; isolate ourselves with our own temp path so
        # propagate() can save its IPP estimate.
        self._tempDir = tempfile.TemporaryDirectory()
        self._origIPPPath = IPPTable.TABLE_PATH
        IPPTable.TABLE_PATH = os.path.join(self._tempDir.name, "ipp.json")

    def tearDown(self):
        IPPTable.TABLE_PATH = self._origIPPPath
        self._tempDir.cleanup()

    @staticmethod
    def _smallSphereScene() -> ScatteringScene:
        outer = Sphere(2.0, order=2, material=ScatteringMaterial(mu_a=0.1, mu_s=10, g=0.85, n=1.37), label="outer")
        inner = Sphere(0.8, order=2, material=ScatteringMaterial(mu_a=0.5, mu_s=5, g=0.85, n=1.37), label="inner")
        grid = Cuboid(5, 5, 5, material=ScatteringMaterial(mu_a=0.05, mu_s=2, g=0.85, n=1.37))
        return ScatteringScene([inner, outer, grid])

    def _runWith(self, useBVH: bool, N: int) -> Stats:
        scene = self._smallSphereScene()
        np.random.seed(0)
        src = PencilPointSource(position=Vector(0, 0, -3), direction=Vector(0, 0, 1), N=N, displaySize=1, seed=0)
        logger = EnergyLogger(scene, defaultBinSize=0.5, keep3D=False)

        orig = cl_photons.CLScene
        cl_photons.CLScene = lambda s, nWU: orig(s, nWU, useBVH=useBVH)
        try:
            src.propagate(scene, logger=logger, showProgress=False)
        finally:
            cl_photons.CLScene = orig

        return Stats(logger)

    def testBVHPropagationMatchesFlatPathOnSphereScene(self):
        # Photon counts kept moderate so the test is fast; the noise floor on absorbance for
        # 5_000 photons is around 1-2 percentage points. We allow 3 pp tolerance.
        N = 5_000
        statsFlat = self._runWith(useBVH=False, N=N)
        statsBVH = self._runWith(useBVH=True, N=N)

        for solidLabel in ("outer", "inner"):
            absFlat = statsFlat.getAbsorbance(solidLabel)
            absBVH = statsBVH.getAbsorbance(solidLabel)
            self.assertAlmostEqual(
                absFlat, absBVH, delta=3.0,
                msg=f"{solidLabel}: BVH absorbance {absBVH:.2f}% differs from flat {absFlat:.2f}% beyond 3pp",
            )
