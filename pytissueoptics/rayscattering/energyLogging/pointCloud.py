from typing import Optional

import numpy as np

from pytissueoptics.scene.logger import Logger, InteractionKey


class PointCloud:
    """ Each point is an array of the form (weight, x, y, z) and the resulting point cloud array is of shape (n, 4). """

    def __init__(self, solidPoints: Optional[np.ndarray] = None,
                 surfacePoints: Optional[np.ndarray] = None):
        self.solidPoints = solidPoints
        self.surfacePoints = surfacePoints

    @property
    def leavingSurfacePoints(self) -> Optional[np.ndarray]:
        if self.surfacePoints is None:
            return None
        return self.surfacePoints[np.where(self.surfacePoints[:, 0] >= 0)[0]]

    @property
    def enteringSurfacePoints(self) -> Optional[np.ndarray]:
        if self.surfacePoints is None:
            return None
        return self.surfacePoints[np.where(self.surfacePoints[:, 0] < 0)[0]]

    @property
    def enteringSurfacePointsPositive(self) -> Optional[np.ndarray]:
        if self.surfacePoints is None:
            return None
        points = self.enteringSurfacePoints
        points[:, 0] = np.negative(points[:, 0])
        return points


class PointCloudFactory:
    def __init__(self, logger: Logger):
        self._logger = logger

    def getPointCloud(self, solidLabel: str = None, surfaceLabel: str = None) -> PointCloud:
        if not solidLabel and not surfaceLabel:
            return PointCloud(self.getPointCloudOfSolids().solidPoints,
                              self.getPointCloudOfSurfaces().surfacePoints)
        points = self._logger.getDataPoints(InteractionKey(solidLabel, surfaceLabel))
        if surfaceLabel:
            return PointCloud(None, points)
        return PointCloud(points, None)

    def getPointCloudOfSolids(self) -> PointCloud:
        points = []
        for solidLabel in self._logger.getStoredSolidLabels():
            solidPoints = self.getPointCloud(solidLabel).solidPoints
            if solidPoints is not None:
                points.append(solidPoints)
        if len(points) == 0:
            return PointCloud(None, None)
        return PointCloud(np.concatenate(points, axis=0), None)

    def getPointCloudOfSurfaces(self, solidLabel: str = None) -> PointCloud:
        points = []
        solidLabels = [solidLabel] if solidLabel else [_solidLabel for _solidLabel in self._logger.getStoredSolidLabels()]
        for _solidLabel in solidLabels:
            for surfaceLabel in self._logger.getStoredSurfaceLabels(_solidLabel):
                points.append(self.getPointCloud(_solidLabel, surfaceLabel).surfacePoints)

        if len(points) == 0:
            return PointCloud(None, None)
        return PointCloud(None, np.concatenate(points, axis=0))
