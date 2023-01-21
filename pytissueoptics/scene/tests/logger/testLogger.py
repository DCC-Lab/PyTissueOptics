import os
import unittest
import tempfile

import numpy as np

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger import Logger, InteractionKey


class TestLogger(unittest.TestCase):
    SOLID_LABEL = "mySolid"
    SURFACE_LABEL = "front"
    INTERACTION_KEY = InteractionKey(SOLID_LABEL, SURFACE_LABEL)

    def testGivenNewLogger_shouldBeEmpty(self):
        logger = Logger()

        self.assertIsNone(logger.getPoints())
        self.assertIsNone(logger.getDataPoints())
        self.assertIsNone(logger.getSegments())

    def testWhenLogNewPoint_shouldAddPointToTheLoggedPoints(self):
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(1, 0, 0), self.INTERACTION_KEY)
        newPoint = Vector(2, 0, 0)

        logger.logPoint(newPoint, self.INTERACTION_KEY)

        self.assertEqual(3, len(logger.getPoints()))
        self.assertTrue(np.array_equal([2, 0, 0], logger.getPoints()[-1]))

    def testWhenLogPointArray_shouldAddAllPointsToTheLoggedPoints(self):
        logger = Logger()
        logger.logPointArray(np.array([[0, 0, 0], [1, 0, 0]]), self.INTERACTION_KEY)

        self.assertEqual(2, len(logger.getPoints()))
        self.assertTrue(np.array_equal([1, 0, 0], logger.getPoints()[-1]))

    def testWhenLogNewDataPoint_shouldAddDataPointToTheLoggedDataPoints(self):
        logger = Logger()
        logger.logDataPoint(-10.5, Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logDataPoint(20.5, Vector(1, 0, 0), self.INTERACTION_KEY)

        logger.logDataPoint(10, Vector(2, 0, 0), self.INTERACTION_KEY)

        self.assertEqual(3, len(logger.getDataPoints()))
        self.assertTrue(np.array_equal([10, 2, 0, 0], logger.getDataPoints()[-1]))

    def testWhenLogDataPointArray_shouldAddAllDataPointsToTheLoggedDataPoints(self):
        logger = Logger()
        logger.logDataPointArray(np.array([[2, 0, 0, 0], [1, 1, 0, 0]]), self.INTERACTION_KEY)

        self.assertEqual(2, len(logger.getDataPoints()))
        self.assertTrue(np.array_equal([1, 1, 0, 0], logger.getDataPoints()[-1]))

    def testWhenLogNewSegment_shouldAddSegmentToTheLoggedSegments(self):
        logger = Logger()
        logger.logSegment(Vector(0, 0, 0), Vector(0, 0, 1), self.INTERACTION_KEY)
        logger.logSegment(Vector(0, 0, 0), Vector(0, 1, 0), self.INTERACTION_KEY)

        logger.logSegment(Vector(0, 0, 0), Vector(1, 0, 0), self.INTERACTION_KEY)

        self.assertEqual(3, len(logger.getSegments()))
        self.assertTrue(np.array_equal([0, 0, 0, 1, 0, 0], logger.getSegments()[-1]))

    def testWhenLogSegmentArray_shouldAddAllSegmentsToTheLoggedSegments(self):
        logger = Logger()
        logger.logSegmentArray(np.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 2, 2, 2]]), self.INTERACTION_KEY)

        self.assertEqual(2, len(logger.getSegments()))
        self.assertTrue(np.array_equal([1, 1, 1, 2, 2, 2], logger.getSegments()[-1]))

    def testWhenLogDataWithEmptyKey_shouldAddData(self):
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), InteractionKey(None, None))
        self.assertEqual(1, len(logger.getPoints()))

    def testWhenLogDataWithNoKey_shouldAddDataWithEmptyKey(self):
        logger = Logger()

        logger.logPoint(Vector(0, 0, 0))

        self.assertEqual(1, len(logger.getPoints(InteractionKey(None, None))))

    def testGivenLoggerWithData_shouldHaveNDataPointsOnlyEqualToTheNumberOfDataPoints(self):
        logger = Logger()
        logger.logDataPoint(10, Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logDataPoint(20, Vector(1, 0, 0), InteractionKey("another solid", "another surface"))
        logger.logPoint(Vector(1, 0, 0), self.INTERACTION_KEY)
        logger.logSegment(Vector(0, 0, 0), Vector(1, 0, 0), self.INTERACTION_KEY)

        self.assertEqual(2, logger.nDataPoints)

    def testWhenGetData_shouldReturnNumpyArrayFormat(self):
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        self.assertTrue(isinstance((logger.getPoints()), np.ndarray))

    def testWhenGetDataWithKey_shouldReturnDataStoredForThisKey(self):
        sameKey = InteractionKey(self.SOLID_LABEL, self.SURFACE_LABEL)
        anotherKey = InteractionKey(self.SOLID_LABEL, "another surface")
        anotherKeyWithoutSurface = InteractionKey(self.SOLID_LABEL)

        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(1, 0, 0), sameKey)
        logger.logPoint(Vector(2, 0, 0), anotherKey)
        logger.logPoint(Vector(3, 0, 0), anotherKeyWithoutSurface)

        self.assertEqual(2, len(logger.getPoints(self.INTERACTION_KEY)))
        self.assertEqual(1, len(logger.getPoints(anotherKeyWithoutSurface)))

    def testWhenGetDataWithSolidLabelKey_shouldOnlyReturnDataStoredWithThisSolidLabelKey(self):
        solidKey = InteractionKey(self.SOLID_LABEL)
        solidSurfaceKey = InteractionKey(self.SOLID_LABEL, self.SURFACE_LABEL)

        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), solidKey)
        logger.logPoint(Vector(1, 0, 0), solidSurfaceKey)

        self.assertEqual(1, len(logger.getPoints(solidKey)))

    def testWhenGetSpecificDataTypeWithEmptyKey_shouldReturnAllDataOfThisType(self):
        logger = Logger()
        logger.logDataPoint(0, Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(0, 0, 0), InteractionKey("another solid", "another surface"))

        self.assertEqual(1, len(logger.getDataPoints()))

    def testWhenGetDataWithEmptyKey_shouldReturnAllData(self):
        anotherKey = InteractionKey(self.SOLID_LABEL, "another surface")
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(1, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(2, 0, 0), anotherKey)
        logger.logPoint(Vector(3, 0, 0), anotherKey)

        self.assertEqual(4, len(logger.getPoints()))
        self.assertEqual(4, len(logger.getPoints(InteractionKey(None, None))))

    def testWhenGetDataWithNonExistentKey_shouldWarnAndReturnNone(self):
        logger = Logger()
        with self.assertWarns(UserWarning):
            self.assertIsNone(logger.getPoints(self.INTERACTION_KEY))

    def testWhenGetDataWithWrongSurfaceLabelKey_shouldWarnAndReturnNone(self):
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)

        with self.assertWarns(UserWarning):
            self.assertIsNone(logger.getPoints(InteractionKey(self.SOLID_LABEL, "wrong surface")))

    def testGivenEmptyLogger_whenGetData_shouldReturnNone(self):
        logger = Logger()

        self.assertIsNone(logger.getPoints())
        self.assertIsNone(logger.getSegments())
        self.assertIsNone(logger.getDataPoints())

    def testWhenGetSolidLabels_shouldReturnAListOfUniqueSolidLabels(self):
        solidLabel1 = "A label"
        solidLabel2 = "Another label"
        interactionKey1 = InteractionKey(solidLabel1, "front")
        interactionKey2 = InteractionKey(solidLabel2)
        interactionKey3 = InteractionKey(solidLabel2, "back")

        logger = Logger()
        aPoint = Vector(0, 0, 0)
        logger.logPoint(aPoint, interactionKey1)
        logger.logPoint(aPoint, interactionKey2)
        logger.logPoint(aPoint, interactionKey3)

        solidLabels = logger.getSeenSolidLabels()

        self.assertEqual(2, len(solidLabels))
        self.assertTrue(solidLabel1 in solidLabels)
        self.assertTrue(solidLabel2 in solidLabels)

    def testWhenGetSurfaceLabels_shouldReturnAListOfUniqueSurfaceLabelsForTheDesiredSolid(self):
        solidLabel1 = "A label"
        solidLabel2 = "Another label"
        surfaceA = "front"
        surfaceB = "back"
        anotherSurface = "top"
        interactionKey1 = InteractionKey(solidLabel1, surfaceA)
        interactionKey2 = InteractionKey(solidLabel1, surfaceB)
        interactionKey3 = InteractionKey(solidLabel1)
        interactionKey4 = InteractionKey(solidLabel2, anotherSurface)

        logger = Logger()
        aPoint = Vector(0, 0, 0)
        logger.logPoint(aPoint, interactionKey1)
        logger.logPoint(aPoint, interactionKey2)
        logger.logPoint(aPoint, interactionKey3)
        logger.logPoint(aPoint, interactionKey4)

        surfaceLabels = logger.getSeenSurfaceLabels(solidLabel1)

        self.assertEqual(2, len(surfaceLabels))
        self.assertTrue(surfaceA in surfaceLabels)
        self.assertTrue(surfaceB in surfaceLabels)

    def testWhenSave_shouldSaveLoggerToFile(self):
        logger = Logger()

        with tempfile.TemporaryDirectory() as tempDir:
            filePath = os.path.join(tempDir, "test.log")
            logger.save(filePath)

            self.assertTrue(os.path.exists(filePath))

    def testWhenSaveWithNoFilePath_shouldWarnAndSaveToDefaultLocation(self):
        logger = Logger()
        with tempfile.TemporaryDirectory() as tempDir:
            logger.DEFAULT_LOGGER_PATH = os.path.join(tempDir, "test.log")

            with self.assertWarns(UserWarning):
                logger.save()

            self.assertTrue(os.path.exists(logger.DEFAULT_LOGGER_PATH))

    def testGivenNewLoggerFromNonExistentFilePath_shouldWarnUserThatNoDataWasLoaded(self):
        with self.assertWarns(UserWarning):
            Logger(fromFilepath="nonExistentFile.log")

    def testGivenLoggerWithFilePath_whenSave_shouldSaveToThisFile(self):
        with tempfile.TemporaryDirectory() as tempDir:
            newFilePath = os.path.join(tempDir, "test.log")
            with self.assertWarns(UserWarning):
                logger = Logger(fromFilepath=newFilePath)

            logger.save()

            self.assertTrue(os.path.exists(newFilePath))

    def testGivenALoggerPreviouslySaved_whenLoad_shouldLoadPreviousLoggerFromFile(self):
        previousLogger = Logger()
        previousLogger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        previousLogger.logPoint(Vector(1, 0, 0), self.INTERACTION_KEY)
        previousLogger.logPoint(Vector(2, 0, 0), self.INTERACTION_KEY)
        previousLogger.info["some key"] = "some metadata"

        with tempfile.TemporaryDirectory() as tempDir:
            filePath = os.path.join(tempDir, "test.log")
            previousLogger.save(filePath)

            logger = Logger()
            logger.load(filePath)

            self.assertTrue(np.array_equal(previousLogger.getPoints(), logger.getPoints()))
            self.assertEqual(previousLogger.info, logger.info)

    def testGivenALoggerPreviouslySaved_whenCreatingNewLoggerFromFile_shouldLoadPreviousLoggerFromFile(self):
        previousLogger = Logger()
        previousLogger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        previousLogger.logPoint(Vector(1, 0, 0), self.INTERACTION_KEY)
        previousLogger.logPoint(Vector(2, 0, 0), self.INTERACTION_KEY)
        previousLogger.info["some key"] = "some metadata"

        with tempfile.TemporaryDirectory() as tempDir:
            filePath = os.path.join(tempDir, "test.log")
            previousLogger.save(filePath)

            logger = Logger(filePath)

            self.assertTrue(np.array_equal(previousLogger.getPoints(), logger.getPoints()))
            self.assertEqual(previousLogger.info, logger.info)
