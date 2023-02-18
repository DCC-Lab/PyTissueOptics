import unittest

import numpy as np

from pytissueoptics.rayscattering.energyLogging import PointCloud


class TestPointCloud(unittest.TestCase):
    def testWhenGetLeavingSurfacePoints_shouldReturnSurfacePointsWhereValueIsPositive(self):
        pointCloud = self._createTestPointCloud()

        leavingSurfacePoints = pointCloud.leavingSurfacePoints

        self.assertEqual(1, len(leavingSurfacePoints))
        self.assertEqual(1, leavingSurfacePoints[0][0])

    def testWhenGetEnteringSurfacePoints_shouldReturnSurfacePointsWhereValueIsNegative(self):
        pointCloud = self._createTestPointCloud()

        enteringSurfacePoints = pointCloud.enteringSurfacePoints

        self.assertEqual(1, len(enteringSurfacePoints))
        self.assertEqual(-1, enteringSurfacePoints[0][0])

    def testWhenGetEnteringSurfacePointsPositive_shouldConvertNegativeValuesToPositive(self):
        pointCloud = self._createTestPointCloud()

        enteringSurfacePointsPositive = pointCloud.enteringSurfacePointsPositive

        self.assertEqual(1, len(enteringSurfacePointsPositive))
        self.assertEqual(1, enteringSurfacePointsPositive[0][0])

    def testGivenEmptyPointCloud_whenGetLeavingSurfacePoints_shouldReturnNone(self):
        pointCloud = PointCloud(None, None)
        leavingSurfacePoints = pointCloud.leavingSurfacePoints
        self.assertIsNone(leavingSurfacePoints)

    def testGivenEmptyPointCloud_whenGetEnteringSurfacePoints_shouldReturnNone(self):
        pointCloud = PointCloud(None, None)
        enteringSurfacePoints = pointCloud.enteringSurfacePoints
        self.assertIsNone(enteringSurfacePoints)

    def testGivenEmptyPointCloud_whenGetEnteringSurfacePointsPositive_shouldReturnNone(self):
        pointCloud = PointCloud(None, None)
        enteringSurfacePointsPositive = pointCloud.enteringSurfacePointsPositive
        self.assertIsNone(enteringSurfacePointsPositive)

    @staticmethod
    def _createTestPointCloud():
        solidPoints = np.array([[0.5, 1, 0, 0], [0.5, 1, 0, 0.1], [0.5, 1, 0, -0.1]])
        surfacePoints = np.array([[1, 1, 0, 0.1], [-1, 1, 0, -0.1]])
        return PointCloud(solidPoints, surfacePoints)

