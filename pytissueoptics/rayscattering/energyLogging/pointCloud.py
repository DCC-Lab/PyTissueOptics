from typing import Optional

import numpy as np


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
