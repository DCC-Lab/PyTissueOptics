from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger, Vector
from pytissueoptics.scene.logger import DataPoint, InteractionKey


@dataclass
class PointCloud:
    solidPoints: Optional[List[DataPoint]] = None
    surfacePoints: Optional[List[DataPoint]] = None


class DisplayConfig:
    """ 3D display configuration dataclass for solid and surface point cloud. """
    def __init__(self, showScene: bool = True,
                 pointSize: float = 0.15, scaleWithValue: bool = True, colormap: str = "rainbow", reverseColormap: bool = False,
                 surfacePointSize: float = 0.01, surfaceScaleWithValue: bool = False, surfaceColormap: str = None, surfaceReverseColormap: bool = None):
        self.showScene = showScene

        self.pointSize = pointSize
        self.scaleWithValue = scaleWithValue
        self.colormap = colormap
        self.reverseColormap = reverseColormap

        self.surfacePointSize = surfacePointSize
        self.surfaceScaleWithValue = surfaceScaleWithValue
        self.surfaceColormap = colormap if surfaceColormap is None else surfaceColormap
        self.surfaceReverseColormap = reverseColormap if surfaceReverseColormap is None else surfaceReverseColormap


class Stats:
    AXES = ["x", "y", "z"]

    def __init__(self, logger: Logger, scene: RayScatteringScene):
        self._logger = logger
        self._scene = scene

    def showEnergy3D(self, solidLabel: str = None, surfaceLabel: str = None, config=DisplayConfig()):
        pointCloud = self.getPointCloud(solidLabel, surfaceLabel)
        return self._show3DPointCloud(pointCloud, config=config)

    def showEnergy3DOfSolids(self, config=DisplayConfig()):
        pointCloud = self._getPointCloudOfSolids()
        return self._show3DPointCloud(pointCloud, config=config)

    def showEnergy3DOfSurfaces(self, solidLabel: str = None, config=DisplayConfig()):
        pointCloud = self._getPointCloudOfSurfaces(solidLabel)
        return self._show3DPointCloud(pointCloud, config=config)

    def _show3DPointCloud(self, pointCloud: PointCloud, config: DisplayConfig):
        from pytissueoptics.scene import MayaviViewer
        viewer = MayaviViewer()

        if config.showScene:
            self._scene.addToViewer(viewer)

        if pointCloud.solidPoints:
            viewer.addDataPoints(pointCloud.solidPoints, scale=config.pointSize,
                                 scaleWithValue=config.scaleWithValue, colormap=config.colormap,
                                 reverseColormap=config.reverseColormap)
        if pointCloud.surfacePoints:
            viewer.addDataPoints(pointCloud.surfacePoints, scale=config.surfacePointSize,
                                 scaleWithValue=config.surfaceScaleWithValue, colormap=config.surfaceColormap,
                                 reverseColormap=config.surfaceReverseColormap)
        viewer.show()

    def getPointCloud(self, solidLabel: str = None, surfaceLabel: str = None) -> PointCloud:
        if not solidLabel and not surfaceLabel:
            return PointCloud(self._getPointCloudOfSolids().solidPoints,
                              self._getPointCloudOfSurfaces().surfacePoints)
        points = self._logger.getDataPoints(InteractionKey(solidLabel, surfaceLabel))
        if surfaceLabel:
            return PointCloud(None, points)
        return PointCloud(points, None)

    def _getPointCloudOfSolids(self) -> PointCloud:
        points = []
        for solidLabel in self._logger.getSolidLabels():
            points.extend(self.getPointCloud(solidLabel).solidPoints)
        return PointCloud(points, None)

    def _getPointCloudOfSurfaces(self, solidLabel: str = None) -> PointCloud:
        points = []
        solidLabels = [solidLabel] if solidLabel else [_solidLabel for _solidLabel in self._logger.getSolidLabels()]
        for _solidLabel in solidLabels:
            for surfaceLabel in self._logger.getSurfaceLabels(_solidLabel):
                points.extend(self.getPointCloud(_solidLabel, surfaceLabel).surfacePoints)
        return PointCloud(None, points)

    def showEnergy2D(self, solidLabel: str = None, surfaceLabel: str = None,
                     projection: Union[str, Vector] = 'y', bins: Union[int, Tuple[int, int]] = None):
        u, v, c = self._get2DScatter(solidLabel, surfaceLabel, projection)

        if bins is not None:
            plt.hist2d(u, v, bins=bins, weights=c)
        else:
            plt.scatter(u, v, c=c)

        uIndex = 0 if projection != 'x' else 1
        vIndex = 1 if projection != 'y' else 2
        plt.xlabel(self.AXES[uIndex])
        plt.ylabel(self.AXES[vIndex])
        plt.show()

        # todo:
        #  - support any projection plane
        #  - add projection of scene interfaces

        # if surfaceLabel : projection = surfaceNormal (mean polygon normal to start)
        # else: default projection to y axis

    def _get2DScatter(self, solidLabel: str = None, surfaceLabel: str = None,
                      projection: Union[str, Vector] = 'y') -> tuple:
        assert projection in self.AXES, 'Projection of arbitrary plane is not supported yet.'

        pointCloud = self.getPointCloud(solidLabel, surfaceLabel)
        if surfaceLabel:
            points = pointCloud.surfacePoints
        else:
            points = pointCloud.solidPoints
        scatter = np.asarray([(p.position.x, p.position.y, p.position.z, p.value) for p in points])

        projectionIndex = self.AXES.index(projection)
        scatter = np.delete(scatter, projectionIndex, axis=1)
        u, v, c = scatter.T
        return u, v, c


# todo: create binned Logger class to bin any logger data dynamically to it (extending)
# binnedLogger.logPoints()
# or logger.bin() ...
