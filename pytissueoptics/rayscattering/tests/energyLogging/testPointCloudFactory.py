import unittest

import numpy as np

from pytissueoptics.rayscattering.energyLogging import PointCloudFactory
from pytissueoptics.scene.logger import Logger, InteractionKey


class TestPointCloudFactory(unittest.TestCase):
    SOLID_LABEL_A = 'solidA'
    SOLID_LABEL_B = 'solidB'
    SOLID_A_SURFACE = 'surfaceA'
    SOLID_B_SURFACE = 'surfaceB'
    N_POINTS_PER_SOLID = 3
    N_POINTS_PER_SURFACE = 2

    def testWhenGetPointCloud_shouldReturnWholePointCloud(self):
        logger = self._createTestLogger()
        pointCloudFactory = PointCloudFactory(logger)
        pointCloud = pointCloudFactory.getPointCloud()

        self.assertEqual(self.N_POINTS_PER_SOLID * 2, len(pointCloud.solidPoints))
        self.assertEqual(self.N_POINTS_PER_SURFACE * 2, len(pointCloud.surfacePoints))

    def testWhenGetPointCloudOfSpecificSolid_shouldOnlyHavePointsInsideSolid(self):
        logger = self._createTestLogger()
        pointCloudFactory = PointCloudFactory(logger)
        pointCloud = pointCloudFactory.getPointCloud(self.SOLID_LABEL_A)

        self.assertEqual(self.N_POINTS_PER_SOLID, len(pointCloud.solidPoints))
        self.assertIsNone(pointCloud.surfacePoints)

    def testWhenGetPointCloudOfSpecificSurface_shouldOnlyHavePointsInsideSurface(self):
        logger = self._createTestLogger()
        pointCloudFactory = PointCloudFactory(logger)
        pointCloud = pointCloudFactory.getPointCloud(self.SOLID_LABEL_A, self.SOLID_A_SURFACE)

        self.assertEqual(self.N_POINTS_PER_SURFACE, len(pointCloud.surfacePoints))
        self.assertIsNone(pointCloud.solidPoints)

    def testWhenGetPointCloudOfSolids_shouldReturnPointCloudWithAllSolidPoints(self):
        logger = self._createTestLogger()
        pointCloudFactory = PointCloudFactory(logger)
        pointCloud = pointCloudFactory.getPointCloudOfSolids()

        self.assertEqual(self.N_POINTS_PER_SOLID * 2, len(pointCloud.solidPoints))
        self.assertIsNone(pointCloud.surfacePoints)

    def testWhenGetPointCloudOfSurfaces_shouldReturnPointCloudWithAllSurfacePoints(self):
        logger = self._createTestLogger()
        pointCloudFactory = PointCloudFactory(logger)
        pointCloud = pointCloudFactory.getPointCloudOfSurfaces()

        self.assertEqual(self.N_POINTS_PER_SURFACE * 2, len(pointCloud.surfacePoints))
        self.assertIsNone(pointCloud.solidPoints)

    def testGivenEmptyLogger_whenGetPointCloud_shouldReturnEmptyPointCloud(self):
        logger = Logger()
        pointCloudFactory = PointCloudFactory(logger)
        pointCloud = pointCloudFactory.getPointCloud()

        self.assertIsNone(pointCloud.solidPoints)
        self.assertIsNone(pointCloud.surfacePoints)

    def _createTestLogger(self):
        logger = Logger()
        solidPointsA = np.array([[0.5, 1, 0, 0], [0.5, 1, 0, 0.1], [0.5, 1, 0, -0.1]])
        solidPointsB = np.array([[0.5, -1, 0, 0], [0.5, -1, 0, 0.1], [0.5, -1, 0, -0.1]])
        surfacePointsA = np.array([[1, 1, 0, 0.1], [-1, 1, 0, -0.1]])
        surfacePointsB = np.array([[1, -1, 0, 0.1], [-1, -1, 0, -0.1]])
        logger.logDataPointArray(solidPointsA, InteractionKey(self.SOLID_LABEL_A))
        logger.logDataPointArray(solidPointsB, InteractionKey(self.SOLID_LABEL_B))
        logger.logDataPointArray(surfacePointsA, InteractionKey(self.SOLID_LABEL_A, self.SOLID_A_SURFACE))
        logger.logDataPointArray(surfacePointsB, InteractionKey(self.SOLID_LABEL_B, self.SOLID_B_SURFACE))
        return logger
