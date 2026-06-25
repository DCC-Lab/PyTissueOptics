import os
import tempfile
import threading
import time
import unittest

from pytissueoptics import (
    Cube,
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
class TestCLPropagationStuckPhotons(unittest.TestCase):
    """
    Regression suite for prior issues with photons that never die: either escaping the
    simulation domain undetected or getting stuck in surface-reflection loops without ever
    scattering between bounces. The CLPhotons orchestrator polls for `weight == 0` photons
    and only advances the outer photonCount counter when one is replaced; a stuck photon
    therefore hangs the entire simulation, even though the kernel itself exits each batch
    on log-buffer exhaustion.

    Each test below builds a scene that historically stressed these failure modes (very
    small geometry, very high scattering coefficient, high refractive-index contrast for
    total internal reflection), runs propagation under a wallclock timeout, and asserts
    completion. Energy-conservation tests then verify that no photons were silently lost.

    Tests are parametrized on useBVH so both intersection backends are covered.
    """

    DEFAULT_TIMEOUT_S = 30.0
    N_PHOTONS = 200

    def setUp(self):
        # See testCLPropagateBVH.py for context: testIPPTable.py leaks IPPTable.TABLE_PATH
        # into class state via a temp dir that's already gone by the time we run.
        self._tempDir = tempfile.TemporaryDirectory()
        self._origIPPPath = IPPTable.TABLE_PATH
        IPPTable.TABLE_PATH = os.path.join(self._tempDir.name, "ipp.json")

    def tearDown(self):
        IPPTable.TABLE_PATH = self._origIPPPath
        self._tempDir.cleanup()

    def _propagateWithTimeout(
        self,
        scene: ScatteringScene,
        source: PencilPointSource,
        logger: EnergyLogger,
        useBVH: bool,
        timeout: float = None,
    ) -> float:
        """Run propagate() on a daemon thread with a wallclock budget.

        On timeout, the daemon keeps running but is leaked for the rest of the process; this
        is acceptable in a test context (the kernel exits each batch via log overflow, so
        nothing is held by the GPU indefinitely). The fail message points future-you at the
        most likely cause."""
        timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT_S
        result = {"done": False, "exc": None, "elapsed": 0.0}

        def _runner():
            origCLScene = cl_photons.CLScene
            cl_photons.CLScene = lambda s, nWU: origCLScene(s, nWU, useBVH=useBVH)
            try:
                t0 = time.time()
                source.propagate(scene, logger=logger, showProgress=False)
                result["elapsed"] = time.time() - t0
                result["done"] = True
            except BaseException as exc:  # noqa: BLE001
                result["exc"] = exc
            finally:
                cl_photons.CLScene = origCLScene

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            self.fail(
                f"Propagation hung beyond {timeout:.0f}s (useBVH={useBVH}). Likely a photon "
                "is stuck in a Fresnel-reflection loop without scattering between hits, so "
                "the orchestrator never observes weight==0 and the outer loop spins forever."
            )
        if result["exc"] is not None:
            raise result["exc"]
        self.assertTrue(result["done"])
        return result["elapsed"]

    # ------- scenes that previously stressed surface-loop and escape failure modes -------

    @staticmethod
    def _smallSphereHighDiffusion() -> ScatteringScene:
        # Radius 0.1 with mu_s=1000 → mean scattering distance is ~0.001, far smaller than
        # the geometry. Photons cross the front surface, bounce around inside, and rely on
        # the scatter-then-reflect-then-scatter sequence to lose weight via interact().
        material = ScatteringMaterial(mu_a=1, mu_s=1000, g=0.9, n=1.4)
        return ScatteringScene([Sphere(0.1, order=2, material=material, label="sphere")])

    @staticmethod
    def _smallCubeHighDiffusion() -> ScatteringScene:
        # Cube faces are aligned with the world axes, so floating-point edge cases at the
        # corners (where three faces meet) historically caused photons to flip-flop between
        # adjacent surfaces without ever scattering.
        material = ScatteringMaterial(mu_a=1, mu_s=1000, g=0.9, n=1.4)
        return ScatteringScene([Cube(0.2, material=material, label="cube")])

    @staticmethod
    def _highIndexTIR() -> ScatteringScene:
        # n=1.6 inside vs 1.0 outside means total internal reflection above ~38 deg incidence.
        # With high g=0.95 (forward-peaked scattering), photons re-hit the surface at similar
        # angles after each scatter; if Fresnel keeps reflecting them, weight only decreases
        # via interact() on each scatter event.
        material = ScatteringMaterial(mu_a=0.5, mu_s=500, g=0.95, n=1.6)
        return ScatteringScene([Sphere(0.15, order=2, material=material, label="tir")])

    def _smallPencilSource(self, zStart: float = -0.2) -> PencilPointSource:
        return PencilPointSource(
            position=Vector(0, 0, zStart),
            direction=Vector(0, 0, 1),
            N=self.N_PHOTONS,
            displaySize=1,
            seed=0,
        )

    # -------------- completion tests: parametrized on (scene, useBVH) --------------------

    def _assertCompletesOnBothPaths(self, sceneFactory, zStart: float = -0.2):
        for useBVH in (False, True):
            scene = sceneFactory()
            source = self._smallPencilSource(zStart=zStart)
            logger = EnergyLogger(scene, defaultBinSize=0.01, keep3D=False)
            elapsed = self._propagateWithTimeout(scene, source, logger, useBVH=useBVH)
            self.assertLess(
                elapsed, self.DEFAULT_TIMEOUT_S,
                msg=f"useBVH={useBVH}: completed but suspiciously slow at {elapsed:.1f}s",
            )

    def testSmallSphereHighDiffusionCompletes(self):
        self._assertCompletesOnBothPaths(self._smallSphereHighDiffusion)

    def testSmallCubeHighDiffusionCompletes(self):
        self._assertCompletesOnBothPaths(self._smallCubeHighDiffusion)

    def testHighIndexContrastTIRCompletes(self):
        self._assertCompletesOnBothPaths(self._highIndexTIR)

    # ----------------- energy-conservation: catches silent photon loss -------------------

    def _runAndGetAbsorbance(self, scene, source, logger, useBVH: bool, label: str) -> float:
        self._propagateWithTimeout(scene, source, logger, useBVH=useBVH)
        return Stats(logger).getAbsorbance(label)

    def testEnergyConservationSmallSphereHighAbsorbance(self):
        # mu_a=10 / mu_t=110 gives an albedo of ~0.91 per scatter, but the geometry is small
        # enough that on average a photon undergoes many scatters before escape. Effective
        # absorbance should approach 100%; a regression that silently drops photons (stack
        # overflow, escape via floating-point bug) would push this floor down.
        for useBVH in (False, True):
            material = ScatteringMaterial(mu_a=10, mu_s=100, g=0.9, n=1.4)
            scene = ScatteringScene([Sphere(0.2, order=2, material=material, label="abs")])
            source = self._smallPencilSource(zStart=-0.3)
            logger = EnergyLogger(scene, defaultBinSize=0.02, keep3D=False)
            absorbance = self._runAndGetAbsorbance(scene, source, logger, useBVH, "abs")
            self.assertGreater(
                absorbance, 80.0,
                msg=(
                    f"useBVH={useBVH}: absorbance {absorbance:.1f}% is below the 80% floor "
                    "for a high-absorption sphere - photons may be escaping or being lost."
                ),
            )

    def testEnergyConservationSmallCubeHighAbsorbance(self):
        # Cube version of the previous test. Cubes have axis-aligned faces, which historically
        # exposed numerical issues at corners that spheres mask.
        for useBVH in (False, True):
            material = ScatteringMaterial(mu_a=10, mu_s=100, g=0.9, n=1.4)
            scene = ScatteringScene([Cube(0.4, material=material, label="abs")])
            source = self._smallPencilSource(zStart=-0.3)
            logger = EnergyLogger(scene, defaultBinSize=0.02, keep3D=False)
            absorbance = self._runAndGetAbsorbance(scene, source, logger, useBVH, "abs")
            self.assertGreater(
                absorbance, 80.0,
                msg=f"useBVH={useBVH}: cube absorbance {absorbance:.1f}% below floor",
            )
