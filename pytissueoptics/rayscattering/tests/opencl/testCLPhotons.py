import unittest

import numpy as np

from pytissueoptics import Cube, EnergyLogger, ScatteringMaterial, ScatteringScene
from pytissueoptics.rayscattering.opencl import OPENCL_OK, WEIGHT_THRESHOLD
from pytissueoptics.rayscattering.opencl.CLPhotons import CLPhotons
from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.logger import InteractionKey


@unittest.skipIf(not OPENCL_OK, "OpenCL device not available.")
class TestCLPhotons(unittest.TestCase):
    def testWhenPropagateWithoutContext_shouldNotPropagate(self):
        positions = np.array([[0, 0, 0], [0, 0, 0]])
        directions = np.array([[0, 0, 1], [0, 0, 1]])
        photons = CLPhotons(positions, directions)

        with self.assertRaises(AssertionError):
            photons.propagate(IPP=10)

    def testWhenPropagate_shouldPropagateUntilAllPhotonsHaveNoMoreEnergy(self):
        N = 100
        # Testing in infinite scene so that photons will scatter all their energy
        worldMaterial = ScatteringMaterial(5, 2, 0.9, 1.4)
        infiniteScene = ScatteringScene([], worldMaterial=worldMaterial)
        logger = EnergyLogger(infiniteScene)

        positions = np.full((N, 3), 0)
        directions = np.full((N, 3), 0)
        directions[:, 2] = 1
        photons = CLPhotons(positions, directions)
        photons.setContext(infiniteScene, Environment(worldMaterial), logger=logger)
        IPP = infiniteScene.getEstimatedIPP(WEIGHT_THRESHOLD)

        photons.propagate(IPP=IPP, verbose=False)

        dataPoints = logger.getRawDataPoints()
        totalWeightScattered = float(np.sum(dataPoints[:, 0]))
        # Roulette effect will result in total weight slightly different from N.
        self.assertAlmostEqual(N, totalWeightScattered, places=1)

    def testWhenPropagateInSolids_shouldLogEnergyWithCorrectInteractionKeys(self):
        N = 100
        material = ScatteringMaterial(5, 2, 0.9, 1.4)
        worldMaterial = ScatteringMaterial()
        cube = Cube(1, material=material, label="cube")
        scene = ScatteringScene([cube], worldMaterial=worldMaterial)
        logger = EnergyLogger(scene)

        positions = np.full((N, 3), 0)
        positions[:, 2] = -1  # start at z = -1, outside of cube starting at z = -0.5
        directions = np.full((N, 3), 0)
        directions[:, 2] = 1
        photons = CLPhotons(positions, directions)
        photons.setContext(scene, Environment(worldMaterial), logger=logger)
        IPP = scene.getEstimatedIPP(WEIGHT_THRESHOLD)

        photons.propagate(IPP=IPP, verbose=False)

        frontSurfacePoints = logger.getRawDataPoints(InteractionKey("cube", "cube_front"))
        energyInput = -np.sum(frontSurfacePoints[:, 0])  # should be around 97% of total energy because of reflections
        cubePoints = logger.getRawDataPoints(InteractionKey("cube"))
        energyScattered = np.sum(cubePoints[:, 0])

        energyLeaving = 0
        for surfaceLabel in logger.getStoredSurfaceLabels("cube"):
            if "front" in surfaceLabel:
                continue
            surfacePoints = logger.getRawDataPoints(InteractionKey("cube", surfaceLabel))
            energyLeaving += np.sum(surfacePoints[:, 0])

        self.assertAlmostEqual(energyInput, energyScattered + energyLeaving, places=2)

    def testWhenPropagateOnly1Photon_shouldPropagate(self):
        N = 1
        # Testing in infinite scene so that photons will scatter all their energy
        worldMaterial = ScatteringMaterial(5, 2, 0.9, 1.4)
        infiniteScene = ScatteringScene([], worldMaterial=worldMaterial)
        logger = EnergyLogger(infiniteScene)

        positions = np.full((N, 3), 0)
        directions = np.full((N, 3), 0)
        directions[:, 2] = 1
        photons = CLPhotons(positions, directions)
        photons.setContext(infiniteScene, Environment(worldMaterial), logger=logger)
        IPP = infiniteScene.getEstimatedIPP(WEIGHT_THRESHOLD)

        photons.propagate(IPP=IPP, verbose=False)

        dataPoints = logger.getRawDataPoints()
        totalWeightScattered = float(np.sum(dataPoints[:, 0]))
        self.assertAlmostEqual(N, totalWeightScattered, places=2)
