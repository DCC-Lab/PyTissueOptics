import copy
import os
import warnings
from typing import Optional, Union, Tuple, List

import matplotlib
matplotlib.use("qt5agg")
import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger, Vector
from pytissueoptics.scene.logger import InteractionKey


class PointCloud:
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


class DisplayConfig:
    """ 3D display configuration dataclass for solid and surface point cloud. """

    def __init__(self, showScene: bool = True, showSource: bool = True, sourceSize: float = 0.1,
                 showPointsAsSpheres: bool = True, pointSize: float = 0.15, scaleWithValue: bool = True,
                 colormap: str = "rainbow", reverseColormap: bool = False, surfacePointSize: float = 0.01,
                 surfaceScaleWithValue: bool = False, surfaceColormap: str = None, surfaceReverseColormap: bool = None):
        self.showScene = showScene
        self.showSource = showSource
        self.sourceSize = sourceSize
        self.showPointsAsSpheres = showPointsAsSpheres

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

        solidSource = source.getEnvironment().solid
        self._sourceSolidLabel = solidSource.getLabel() if solidSource else None
        self._source = source

    def showEnergy3D(self, solidLabel: str = None, surfaceLabel: str = None, config=DisplayConfig()):
        pointCloud = self.getPointCloud(solidLabel, surfaceLabel)
        return self._show3DPointCloud(pointCloud, config=config)

    def showEnergy3DOfSolids(self, config=DisplayConfig()):
        pointCloud = self._getPointCloudOfSolids()
        return self._show3DPointCloud(pointCloud, config=config)

    def showEnergy3DOfSurfaces(self, solidLabel: str = None, config=DisplayConfig()):
        pointCloud = self._getPointCloudOfSurfaces(solidLabel)
        return self._show3DPointCloud(pointCloud, config=config)

    def getPhotonCount(self) -> int:
        if "photonCount" not in self._logger.info:
            return self._source.getPhotonCount()
        return self._logger.info["photonCount"]

    def _show3DPointCloud(self, pointCloud: PointCloud, config: DisplayConfig):
        from pytissueoptics.scene import MayaviViewer, MAYAVI_AVAILABLE
        if not MAYAVI_AVAILABLE:
            warnings.warn("Package 'mayavi' is not available. Please install it to use 3D visualizations.")
            return

        viewer = MayaviViewer()

        if config.showScene:
            if self._scene is None:
                warnings.warn("Cannot display Scene objects when no scene was provided to the Stats.")
            else:
                self._scene.addToViewer(viewer)

        if config.showSource:
            self._source.addToViewer(viewer, size=config.sourceSize)

        if pointCloud.solidPoints is not None:
            viewer.addDataPoints(pointCloud.solidPoints, scale=config.pointSize,
                                 scaleWithValue=config.scaleWithValue, colormap=config.colormap,
                                 reverseColormap=config.reverseColormap, asSpheres=config.showPointsAsSpheres)
        if pointCloud.surfacePoints is not None:
            viewer.addDataPoints(pointCloud.leavingSurfacePoints, scale=config.surfacePointSize,
                                 scaleWithValue=config.surfaceScaleWithValue, colormap=config.surfaceColormap,
                                 reverseColormap=config.surfaceReverseColormap, asSpheres=config.showPointsAsSpheres)
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
            solidPoints = self.getPointCloud(solidLabel).solidPoints
            if solidPoints is not None:
                points.append(solidPoints)
        if len(points) == 0:
            return PointCloud(None, None)
        return PointCloud(np.concatenate(points, axis=0), None)

    def _getPointCloudOfSurfaces(self, solidLabel: str = None) -> PointCloud:
        points = []
        solidLabels = [solidLabel] if solidLabel else [_solidLabel for _solidLabel in self._logger.getSolidLabels()]
        for _solidLabel in solidLabels:
            for surfaceLabel in self._logger.getSurfaceLabels(_solidLabel):
                points.append(self.getPointCloud(_solidLabel, surfaceLabel).surfacePoints)

        if len(points) == 0:
            return PointCloud(None, None)
        return PointCloud(None, np.concatenate(points, axis=0))

    def showEnergy2D(self, solidLabel: str = None, surfaceLabel: str = None,
                     projection: Union[str, Vector] = 'y', bins: Union[int, Tuple[int, int]] = None,
                     limits: List[List[float]] = None, logScale: bool = False, enteringSurface=False,
                     colormap: str = 'viridis'):
        u, v, c = self._get2DScatter(solidLabel, surfaceLabel, projection, enteringSurface)

        norm = matplotlib.colors.LogNorm() if logScale else None
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        if bins is not None:
            plt.hist2d(u, v, bins=bins, weights=c, range=limits, norm=norm, cmap=cmap)
        else:
            plt.scatter(u, v, c=c, norm=norm, cmap=cmap)

        uIndex = 0 if projection != 'x' else 2
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
        u, v, c = [scatter[:, i] for i in range(4) if i != projectionIndex]
        return u, v, c

    def _get3DScatter(self, solidLabel: str = None, surfaceLabel: str = None, enteringSurface=False):
        pointCloud = self.getPointCloud(solidLabel, surfaceLabel)
        if surfaceLabel and enteringSurface:
            points = pointCloud.enteringSurfacePoints
            points[:, :1] = -points[:, :1]
        elif surfaceLabel:
            points = pointCloud.leavingSurfacePoints
        else:
            points = pointCloud.solidPoints

        if points is None:
            return np.empty(0)
        scatter = np.concatenate([points[:, 1:], points[:, :1]], axis=1)
        return scatter

    def showEnergy1D(self, solidLabel: str = None, surfaceLabel: str = None, along: str = 'z', bins: int = None,
                     limits: List[float] = None):
        x, c = self._get1DScatter(solidLabel, surfaceLabel, along)
        if bins is not None:
            plt.hist(x, bins=bins, weights=c, range=limits)
        else:
            plt.scatter(x, c)
        plt.xlabel(along)
        plt.ylabel('Energy')
        plt.show()

    def _get1DScatter(self, solidLabel: str = None, surfaceLabel: str = None, along: str = 'z') -> tuple:
        assert along in self.AXES, 'Projection of arbitrary plane is not supported yet.'
        alongIndex = self.AXES.index(along)

        scatter = self._get3DScatter(solidLabel, surfaceLabel)
        if len(scatter) == 0:
            return [], []
        x, c = scatter[:, alongIndex], scatter[:, -1]
        return x, c

    def getAbsorbance(self, solidLabel: str = None, useTotalEnergy=False) -> float:
        points = self.getPointCloud(solidLabel).solidPoints
        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return self._sumEnergy(points) / energyInput

    def _getEnergyInput(self, solidLabel: str = None) -> float:
        if solidLabel is None:
            return self.getPhotonCount()
        points = self._getPointCloudOfSurfaces(solidLabel).enteringSurfacePoints
        energy = self._sumEnergy(points)

        if self._sourceSolidLabel == solidLabel:
            energy += self.getPhotonCount()
        return energy

    def getTransmittance(self, solidLabel: str = None, surfaceLabel: str = None, useTotalEnergy=False):
        """ Uses local energy input for the desired solid by default. Specify 'useTotalEnergy' = True
        to compare instead with total input energy of the scene. """
        if surfaceLabel is None:
            points = self._getPointCloudOfSurfaces(solidLabel).leavingSurfacePoints
        else:
            points = self.getPointCloud(solidLabel, surfaceLabel).leavingSurfacePoints

        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return self._sumEnergy(points) / energyInput

    @staticmethod
    def _sumEnergy(points: np.ndarray):
        return np.abs(np.sum(points[:, 0])) if points is not None else 0

    def _makeReport(self, solidLabel: str = None, reportString: str = ""):
        if solidLabel:
            reportString += self._reportSolid(solidLabel)
        else:
            for solidLabel in self._logger.getSolidLabels():
                reportString = self._makeReport(solidLabel, reportString)
        return reportString

    def report(self, solidLabel: str = None, saveToFile: str = None, verbose=True):
        reportString = self._makeReport(solidLabel=solidLabel)
        if saveToFile:
            self.saveReport(reportString, saveToFile)
        if verbose:
            print(reportString)

    def _reportSolid(self, solidLabel: str):
        reportString = "Report of solid '{}'\n".format(solidLabel)
        try:
            reportString += (
                "  Absorbance: {:.1f}% ({:.1f}% of total power)\n".format(100 * self.getAbsorbance(solidLabel),
                                                                          100 * self.getAbsorbance(solidLabel,
                                                                                                   useTotalEnergy=True)))
            reportString += ("  Absorbance + Transmittance: {:.1f}%\n".format(100 * (self.getAbsorbance(solidLabel) +
                                                                                     self.getTransmittance(
                                                                                         solidLabel))))

            for surfaceLabel in self._logger.getSurfaceLabels(solidLabel):
                transmittance = "{0:.1f}".format(100 * self.getTransmittance(solidLabel, surfaceLabel))
                reportString += f"    Transmittance at '{surfaceLabel}': {transmittance}%\n"

        except ZeroDivisionError:
            warnings.warn("No energy input for solid '{}'".format(solidLabel))
            reportString += ("  Absorbance: N/A ({:.1f}% of total power)\n".format(100 * self.getAbsorbance(solidLabel,
                                                                                                            useTotalEnergy=True)))
            reportString += "  Absorbance + Transmittance: N/A\n"
        return reportString

    @staticmethod
    def saveReport(report: str, filepath: str = None):
        if filepath is None:
            filepath = "simulation_report"
            warnings.warn(f"No filepath specified. Saving to {filepath}.")
        i = 0
        filename, extension = filepath.split(".")
        if extension == "":
            extension = "txt"
        if os.path.exists(filepath):
            while os.path.exists("{}_{}.{}".format(filepath, i, extension)):
                i += 1
            filename = "{}_{}".format(filepath, i)
        filepath = "{}.{}".format(filename, extension)
        with open(filepath, "wb") as file:
            file.write(report.encode("utf-8"))
            file.close()

    def getAverageInteractionsPerPhoton(self):
        return len(self._logger.getDataPoints()) / self.getPhotonCount()
