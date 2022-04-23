import copy
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering.source import Source
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

    def __init__(self, logger: Logger, scene: RayScatteringScene, source: Source):
        self._logger = logger
        self._scene = scene
        self._photonCount = source.photonCount

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
                     projection: Union[str, Vector] = 'y', bins: Union[int, Tuple[int, int]] = None,
                     logScale: bool = False, colormap: str = 'viridis'):
        u, v, c = self._get2DScatter(solidLabel, surfaceLabel, projection)

        norm = matplotlib.colors.LogNorm() if logScale else None
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        if bins is not None:
            plt.hist2d(u, v, bins=bins, weights=c, norm=norm, cmap=cmap)
        else:
            plt.scatter(u, v, c=c, norm=norm, cmap=cmap)

        uIndex = 0 if projection != 'x' else 1
        vIndex = 1 if projection != 'y' else 2
        plt.xlabel(self.AXES[uIndex])
        plt.ylabel(self.AXES[vIndex])
        plt.show()

        # todo:
        #  - support any projection plane (if surfaceLabel : projection = surfaceNormal (mean polygon normal to start))
        #  - add projection of scene interfaces

    def _get2DScatter(self, solidLabel: str = None, surfaceLabel: str = None,
                      projection: Union[str, Vector] = 'y') -> tuple:
        assert projection in self.AXES, 'Projection of arbitrary plane is not supported yet.'
        projectionIndex = self.AXES.index(projection)

        scatter = self._getScatter(solidLabel, surfaceLabel)
        u, v, c = np.delete(scatter, projectionIndex, axis=0)
        return u, v, c

    def _getScatter(self, solidLabel: str = None, surfaceLabel: str = None):
        pointCloud = self.getPointCloud(solidLabel, surfaceLabel)
        if surfaceLabel:
            points = pointCloud.surfacePoints
        else:
            points = pointCloud.solidPoints
        scatter = np.asarray([(p.position.x, p.position.y, p.position.z, p.value) for p in points])
        return scatter.T

    def showEnergy1D(self, solidLabel: str = None, surfaceLabel: str = None, along: str = 'z', bins: int = None):
        x, c = self._get1DScatter(solidLabel, surfaceLabel, along)
        if bins is not None:
            plt.hist(x, bins=bins, weights=c)
        else:
            plt.scatter(x, c)
        plt.xlabel(along)
        plt.ylabel('Energy')
        plt.show()

    def _get1DScatter(self, solidLabel: str = None, surfaceLabel: str = None, along: str = 'z') -> tuple:
        assert along in self.AXES, 'Projection of arbitrary plane is not supported yet.'
        alongIndex = self.AXES.index(along)

        scatter = self._getScatter(solidLabel, surfaceLabel)
        x, c = scatter[alongIndex], scatter[-1]
        return x, c

    def getAbsorbance(self, solidLabel: str = None) -> float:
        points = self.getPointCloud(solidLabel).solidPoints
        energy = sum([p.value for p in points])
        return energy / self._photonCount

    def getTransmittance(self, solidLabel: str = None, surfaceLabel: str = None):
        # fixme: transmittance is wrong for now since we don't discriminate between
        #   energy that entered or left the surface (easy fix would be to discriminate
        #   during logging by logging negative energy if the photon entered the surface)
        if surfaceLabel is None:
            points = self._getPointCloudOfSurfaces(solidLabel).surfacePoints
        else:
            points = self.getPointCloud(solidLabel, surfaceLabel).surfacePoints

        energy = sum([p.value for p in points])
        return energy / self._photonCount

    def report(self, solidLabel: str = None):
        if solidLabel:
            return self._reportSolid(solidLabel)
        for solidLabel in self._logger.getSolidLabels():
            self.report(solidLabel)

    def _reportSolid(self, solidLabel: str):
        print("Report of solid '{}'".format(solidLabel))
        print("  Absorbance: {0:.1f}% of total power. ".format(100 * self.getAbsorbance(solidLabel)))
        for surfaceLabel in self._logger.getSurfaceLabels(solidLabel):
            transmittance = "{0:.1f}".format(100 * self.getTransmittance(solidLabel, surfaceLabel))
            print(f"  Transmittance at '{surfaceLabel}': {transmittance}% of total power.")


# todo: create binned Logger class to bin any logger data dynamically to it (extending)
# binnedLogger.logPoints()
# or logger.bin() ...
