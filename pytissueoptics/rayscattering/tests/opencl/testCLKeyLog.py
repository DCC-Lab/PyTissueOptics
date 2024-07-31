import envtest
import unittest

import numpy as np
from mockito import mock, when, verify, arg_that

from pytissueoptics import Cube, ScatteringMaterial, Sphere, ScatteringScene, EnergyLogger
from pytissueoptics.rayscattering.opencl.utils import CLKeyLog
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.CLScene import CLScene, NO_LOG_ID, NO_SOLID_ID, NO_SURFACE_ID, NO_SOLID_LABEL
from pytissueoptics.scene.logger import InteractionKey


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLKeyLog(unittest.TestCase):
    def setUp(self):
        material1 = ScatteringMaterial(2, 0.8, 0.8, 1.4)
        material2 = ScatteringMaterial(5, 0.5, 0.8, 1.4)
        self.cube = Cube(4, material=material1, label="cube")
        self.sphere = Sphere(1, material=material2, label="sphere")
        self.scene = ScatteringScene([self.cube, self.sphere])

    def _createTestLog(self, sceneCL):
        cubeID = sceneCL.getSolidID(self.cube)
        cubeSurfaceIDs = sceneCL.getSurfaceIDs(cubeID)
        sphereID = sceneCL.getSolidID(self.sphere)
        sphereSurfaceIDs = sceneCL.getSurfaceIDs(sphereID)

        log = np.array([[0, 0, 0, 0, cubeID, NO_SURFACE_ID],
                        [2, 0, 0, 0, sphereID, NO_SURFACE_ID],
                        [3, 0, 0, 0, NO_SOLID_ID, NO_SURFACE_ID],
                        [4, 9, 9, 9, NO_LOG_ID, 9],
                        [5, 0, 0, 0, cubeID, cubeSurfaceIDs[1]],
                        [6, 0, 0, 0, sphereID, sphereSurfaceIDs[1]],
                        [1, 0, 0, 0, cubeID, NO_SURFACE_ID]])
        return log

    def testGivenCLKeyLog_whenTransferToSceneLogger_shouldLogDataWithInteractionKeys(self):
        sceneCL = CLScene(self.scene, nWorkUnits=10)
        log = self._createTestLog(sceneCL)
        clKeyLog = CLKeyLog(log, sceneCL)
        sceneLogger = mock(EnergyLogger)
        when(sceneLogger).logDataPointArray(...).thenReturn()

        clKeyLog.toSceneLogger(sceneLogger)

        verify(sceneLogger, times=5).logDataPointArray(...)

        expectedCubeData = arg_that(lambda arg: np.array_equal(arg, np.array([[0, 0, 0, 0], [1, 0, 0, 0]])))
        verify(sceneLogger).logDataPointArray(expectedCubeData, InteractionKey(self.cube.getLabel()))

        expectedValuesWithKeys = [(2, InteractionKey(self.sphere.getLabel())),
                                  (3, InteractionKey(NO_SOLID_LABEL)),
                                  (5, InteractionKey(self.cube.getLabel(), self.cube.surfaceLabels[0])),
                                  (6, InteractionKey(self.sphere.getLabel(), self.sphere.surfaceLabels[0]))]
        for value, expectedKey in expectedValuesWithKeys:
            expectedData = arg_that(lambda arg: np.array_equal(arg, np.array([[value, 0, 0, 0]])))
            verify(sceneLogger).logDataPointArray(expectedData, expectedKey)

    def testGivenCLKeyLogForInfiniteScene_whenTransferToSceneLogger_shouldLogDataWithInteractionKeys(self):
        self.scene = ScatteringScene([], worldMaterial=ScatteringMaterial(1, 0.8, 0.8, 1.4))
        sceneCL = CLScene(self.scene, nWorkUnits=10)
        log = np.array([[1, 0, 0, 0, NO_SOLID_ID, NO_SURFACE_ID],
                        [2, 0, 0, 0, NO_LOG_ID, 99],
                        [3, 0, 0, 0, NO_SOLID_ID, NO_SURFACE_ID]])
        clKeyLog = CLKeyLog(log, sceneCL)
        sceneLogger = mock(EnergyLogger)
        when(sceneLogger).logDataPointArray(...).thenReturn()

        clKeyLog.toSceneLogger(sceneLogger)

        verify(sceneLogger, times=1).logDataPointArray(...)
        expectedWorldData = arg_that(lambda arg: np.array_equal(arg, np.array([[1, 0, 0, 0], [3, 0, 0, 0]])))
        verify(sceneLogger).logDataPointArray(expectedWorldData, InteractionKey(NO_SOLID_LABEL))

if __name__ == "__main__":
    unittest.main()
