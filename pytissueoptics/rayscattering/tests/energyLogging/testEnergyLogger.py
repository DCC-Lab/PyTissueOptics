import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.scene.solids import Cube
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger import InteractionKey


class TestEnergyLogger(unittest.TestCase):
    CUBE_CENTER = Vector(0.5, 0.5, 0.5)
    CUBE = Cube(1, position=CUBE_CENTER, material=ScatteringMaterial(), label='cube')
    TEST_SCENE = ScatteringScene([CUBE])
    INTERACTION_KEY = InteractionKey(CUBE.getLabel())

    def setUp(self):
        self.logger = EnergyLogger(self.TEST_SCENE)

    def testShouldBeEmpty(self):
        self.assertTrue(self.logger.isEmpty)

    def testShouldHaveDefault2DViews(self):
        N_SCENE_VIEWS = 3
        N_CUBE_VIEWS = 3
        N_VIEWS_PER_SURFACE = 2
        N_SURFACES = 6
        expectedNumberOfViews = N_SCENE_VIEWS + N_CUBE_VIEWS + N_SURFACES * N_VIEWS_PER_SURFACE

        self.assertEqual(expectedNumberOfViews, len(self.logger.views))

    def testWhenGetView_shouldReturnView(self):
        self.assertEqual(self.logger.views[5], self.logger.getView(5))

    def testWhenGetViewWithInvalidIndex_shouldRaiseException(self):
        with self.assertRaises(IndexError):
            self.logger.getView(25)

    @patch('sys.stdout', new_callable=io.StringIO)
    def testWhenListViews_shouldPrintListOfViews(self, mock_stdout):
        self.logger.listViews()
        self.assertEqual(len(self.logger.views) + 1, len(mock_stdout.getvalue().splitlines()))

    def testWhenLogDataPointArray_shouldNotBeEmpty(self):
        self.logger.logDataPointArray(np.array([[0.5, 0, 0, 0]]), self.INTERACTION_KEY)
        self.assertFalse(self.logger.isEmpty)

    def testWhenLogDataPoint_shouldNotBeEmpty(self):
        self.logger.logDataPoint(0.5, Vector(0, 0, 0), self.INTERACTION_KEY)
        self.assertFalse(self.logger.isEmpty)

    def testGivenLoggerWithData_whenUpdateView_shouldExtractDataToTheView(self):
        self.logger.logDataPoint(0.5, self.CUBE_CENTER, self.INTERACTION_KEY)
        cubeViewZ = self.logger.views[5]

        self.logger.updateView(cubeViewZ)

        self.assertEqual(0.5, cubeViewZ.getSum())

    def testWhenSave_shouldSaveLoggerToFile(self):
        with tempfile.TemporaryDirectory() as tempDir:
            filePath = os.path.join(tempDir, "test.log")
            self.logger.save(filePath)

            self.assertTrue(os.path.exists(filePath))

    def testWhenSaveWithNoFilePath_shouldWarnAndSaveToDefaultLocation(self):
        with tempfile.TemporaryDirectory() as tempDir:
            self.logger.DEFAULT_LOGGER_PATH = os.path.join(tempDir, "test.log")

            with self.assertWarns(UserWarning):
                self.logger.save()

            self.assertTrue(os.path.exists(self.logger.DEFAULT_LOGGER_PATH))

    def testGivenNewLoggerFromNonExistentFilePath_shouldWarnUserThatNoDataWasLoaded(self):
        with self.assertWarns(UserWarning):
            EnergyLogger(self.TEST_SCENE, filepath="nonExistentFile.log")

    def testGivenLoggerWithFilePath_whenSave_shouldSaveToThisFile(self):
        with tempfile.TemporaryDirectory() as tempDir:
            newFilePath = os.path.join(tempDir, "test.log")
            with self.assertWarns(UserWarning):
                logger = EnergyLogger(self.TEST_SCENE, filepath=newFilePath)

            logger.save()

            self.assertTrue(os.path.exists(newFilePath))

    def testGivenALoggerPreviouslySaved_whenLoad_shouldLoadPreviousLoggerFromFile(self):
        previousLogger = EnergyLogger(self.TEST_SCENE)
        previousLogger.logDataPointArray(np.array([[0.5, 0, 0, 0]]), self.INTERACTION_KEY)
        previousLogger.info["some key"] = "some metadata"

        with tempfile.TemporaryDirectory() as tempDir:
            filePath = os.path.join(tempDir, "test.log")
            previousLogger.save(filePath)

            logger = EnergyLogger(self.TEST_SCENE)
            logger.load(filePath)

            self.assertTrue(np.array_equal(previousLogger.getDataPoints(), logger.getDataPoints()))
            self.assertEqual(previousLogger.info, logger.info)

    def testGivenALoggerPreviouslySaved_whenCreatingNewLoggerFromFile_shouldLoadPreviousLoggerFromFile(self):
        previousLogger = EnergyLogger(self.TEST_SCENE)
        previousLogger.logDataPointArray(np.array([[0.5, 0, 0, 0]]), self.INTERACTION_KEY)
        previousLogger.info["some key"] = "some metadata"

        with tempfile.TemporaryDirectory() as tempDir:
            filePath = os.path.join(tempDir, "test.log")
            previousLogger.save(filePath)

            logger = EnergyLogger(self.TEST_SCENE, filePath)

            self.assertTrue(np.array_equal(previousLogger.getDataPoints(), logger.getDataPoints()))
            self.assertEqual(previousLogger.info, logger.info)

    def testWhenLogPointArray_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.logger.logPointArray(np.array([[0, 0, 0]]), self.INTERACTION_KEY)

    def testWhenLogPoint_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)

    def testWhenLogSegmentArray_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.logger.logSegmentArray(np.array([[0, 0, 0, 0, 0, 0]]), self.INTERACTION_KEY)

    def testWhenLogSegment_shouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.logger.logSegment(Vector(0, 0, 0), Vector(0, 0, 0), self.INTERACTION_KEY)

# todo: probably create a test skeleton
#  and run for both 2D and 3D logger
