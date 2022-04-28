import copy
import warnings
from typing import List, Optional, Union, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger, Vector
from pytissueoptics.scene.logger import DataPoint, InteractionKey


class PointCloud:
    def __init__(self, solidPoints: Optional[List[DataPoint]] = None,
                 surfacePoints: Optional[List[DataPoint]] = None):
        self.solidPoints = solidPoints if solidPoints is not None else []
        self.surfacePoints = surfacePoints if surfacePoints is not None else []

    @property
    def leavingSurfacePoints(self) -> List[DataPoint]:
        return [p for p in self.surfacePoints if p.value >= 0]

    @property
    def enteringSurfacePoints(self) -> List[DataPoint]:
        return [p for p in self.surfacePoints if p.value < 0]


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

    def __init__(self, logger: Logger, source: Source, scene: RayScatteringScene = None):
        self._logger = logger
        self._scene = scene
        self._photonCount = source.getPhotonCount()

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
            if self._scene is None:
                warnings.warn("Cannot display Scene objects when no scene was provided to the Stats.")
            else:
                self._scene.addToViewer(viewer)

        if pointCloud.solidPoints:
            viewer.addDataPoints(pointCloud.solidPoints, scale=config.pointSize,
                                 scaleWithValue=config.scaleWithValue, colormap=config.colormap,
                                 reverseColormap=config.reverseColormap)
        if pointCloud.surfacePoints:
            viewer.addDataPoints(pointCloud.leavingSurfacePoints, scale=config.surfacePointSize,
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
                     logScale: bool = False, enteringSurface=False, colormap: str = 'viridis'):
        u, v, c = self._get2DScatter(solidLabel, surfaceLabel, projection, enteringSurface)

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

    def _get2DScatter(self, solidLabel: str = None, surfaceLabel: str = None,
                      projection: Union[str, Vector] = 'y', enteringSurface=False) -> tuple:
        assert projection in self.AXES, 'Projection of arbitrary plane is not supported yet.'
        projectionIndex = self.AXES.index(projection)

        scatter = self._get3DScatter(solidLabel, surfaceLabel, enteringSurface=enteringSurface)
        if len(scatter) == 0:
            return [], [], []
        u, v, c = np.delete(scatter, projectionIndex, axis=0)
        return u, v, c

    def _get3DScatter(self, solidLabel: str = None, surfaceLabel: str = None, enteringSurface=False):
        pointCloud = self.getPointCloud(solidLabel, surfaceLabel)
        if surfaceLabel and enteringSurface:
            points = pointCloud.enteringSurfacePoints
            for p in points:
                p.value = -p.value
        elif surfaceLabel:
            points = pointCloud.leavingSurfacePoints
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

        scatter = self._get3DScatter(solidLabel, surfaceLabel)
        x, c = scatter[alongIndex], scatter[-1]
        return x, c

    def getAbsorbance(self, solidLabel: str = None, useTotalEnergy=False) -> float:
        points = self.getPointCloud(solidLabel).solidPoints
        energy = sum([p.value for p in points])
        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self._photonCount
        return energy / energyInput

    def _getEnergyInput(self, solidLabel: str = None) -> float:
        if solidLabel is None:
            return self._photonCount
        points = self._getPointCloudOfSurfaces(solidLabel).enteringSurfacePoints
        energy = -sum([p.value for p in points])
        return energy

    def getTransmittance(self, solidLabel: str = None, surfaceLabel: str = None, useTotalEnergy=False):
        """ Uses local energy input for the desired solid by default. Specify 'useTotalEnergy' = True
        to compare instead with total input energy of the scene. """
        # fixme: transmittance is wrong for cuboid stacks since each stacked layer loses reference to its
        #  previous interface (only base layer will be complete)
        if surfaceLabel is None:
            points = self._getPointCloudOfSurfaces(solidLabel).leavingSurfacePoints
        else:
            points = self.getPointCloud(solidLabel, surfaceLabel).leavingSurfacePoints

        energy = sum([p.value for p in points])
        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self._photonCount
        return energy / energyInput

    def report(self, solidLabel: str = None):
        if solidLabel:
            return self._reportSolid(solidLabel)
        for solidLabel in self._logger.getSolidLabels():
            self.report(solidLabel)

    def _reportSolid(self, solidLabel: str):
        print("Report of solid '{}'".format(solidLabel))
        print("  Absorbance: {:.1f}% ({:.1f}% of total power)".format(100 * self.getAbsorbance(solidLabel),
              100 * self.getAbsorbance(solidLabel, useTotalEnergy=True)))
        print("  Absorbance + Transmittance: {:.1f}%".format(100 * (self.getAbsorbance(solidLabel) +
                                                                    self.getTransmittance(solidLabel))))
        for surfaceLabel in self._logger.getSurfaceLabels(solidLabel):
            transmittance = "{0:.1f}".format(100 * self.getTransmittance(solidLabel, surfaceLabel))
            print(f"    Transmittance at '{surfaceLabel}': {transmittance}%")
