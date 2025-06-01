from typing import Tuple

import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.display.profiles import Profile1D
from pytissueoptics.rayscattering.display.utils import Direction
from pytissueoptics.rayscattering.energyLogging import EnergyLogger, EnergyType, PointCloudFactory
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene


class ProfileFactory:
    def __init__(self, scene: ScatteringScene, logger: EnergyLogger):
        self._scene = scene
        self._logger = logger
        self._pointCloudFactory = PointCloudFactory(logger)

        self._defaultBinSize3D = logger.defaultBinSize
        if isinstance(self._defaultBinSize3D, float):
            self._defaultBinSize3D = [logger.defaultBinSize] * 3
        self._infiniteLimits = logger.infiniteLimits

    def create(
        self,
        horizontalDirection: Direction,
        solidLabel: str = None,
        surfaceLabel: str = None,
        surfaceEnergyLeaving: bool = True,
        limits: Tuple[float, float] = None,
        binSize: float = None,
        energyType=EnergyType.DEPOSITION,
    ) -> Profile1D:
        solidLabel, surfaceLabel = self._correctCapitalization(solidLabel, surfaceLabel)
        if binSize is None:
            binSize = self._defaultBinSize3D[horizontalDirection.axis]
        if limits is None:
            limits = self._getDefaultLimits(horizontalDirection, solidLabel)
        limits = (min(limits), max(limits))
        bins = int((limits[1] - limits[0]) / binSize)

        if self._logger.has3D:
            histogram = self._extractHistogramFrom3D(
                horizontalDirection, solidLabel, surfaceLabel, surfaceEnergyLeaving, limits, bins, energyType
            )
        else:
            histogram = self._extractHistogramFromViews(
                horizontalDirection, solidLabel, surfaceLabel, surfaceEnergyLeaving, limits, bins, energyType
            )

        name = self._createName(horizontalDirection, solidLabel, surfaceLabel, surfaceEnergyLeaving, energyType)
        return Profile1D(histogram, horizontalDirection, limits, name, energyType)

    def _getDefaultLimits(self, horizontalDirection: Direction, solidLabel: str = None):
        if solidLabel:
            solid = self._scene.getSolid(solidLabel)
            limits3D = solid.getBoundingBox().xyzLimits
        else:
            sceneBoundingBox = self._scene.getBoundingBox()
            if sceneBoundingBox is None:
                limits3D = self._infiniteLimits
            else:
                limits3D = sceneBoundingBox.xyzLimits
        limits3D = [(d[0], d[1]) for d in limits3D]
        return limits3D[horizontalDirection.axis]

    def _extractHistogramFrom3D(
        self,
        horizontalDirection: Direction,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool,
        limits: Tuple[float, float],
        bins: int,
        energyType: EnergyType,
    ):
        pointCloud = self._pointCloudFactory.getPointCloud(solidLabel, surfaceLabel, energyType=energyType)

        if surfaceLabel:
            if surfaceEnergyLeaving:
                dataPoints = pointCloud.leavingSurfacePoints
            else:
                dataPoints = pointCloud.enteringSurfacePointsPositive
        else:
            dataPoints = pointCloud.solidPoints
        if dataPoints is None:
            return np.zeros(bins)

        x, w = dataPoints[:, horizontalDirection.axis + 1], dataPoints[:, 0]
        histogram, _ = np.histogram(x, bins=bins, range=limits, weights=w)
        return histogram

    def _extractHistogramFromViews(
        self,
        horizontalDirection: Direction,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool,
        limits: Tuple[float, float],
        bins: int,
        energyType: EnergyType,
    ):
        energyMismatch = False

        for view in self._logger.views:
            if view.axis == horizontalDirection.axis:
                continue
            if not utils.labelsEqual(view.solidLabel, solidLabel):
                continue
            if not utils.labelsEqual(view.surfaceLabel, surfaceLabel):
                continue
            if surfaceLabel:
                if view.surfaceEnergyLeaving != surfaceEnergyLeaving:
                    continue

            if view.axisU == horizontalDirection.axis:
                viewLimits = view.limitsU
                viewBins = view.binsU
            else:
                viewLimits = view.limitsV
                viewBins = view.binsV
            if sorted(viewLimits) != sorted(limits):
                # todo: allow contained limits
                continue
            if viewBins != bins:
                continue
            if not surfaceLabel and energyType != view.energyType:
                energyMismatch = True
                continue

            return self._extractHistogramFromView(view, horizontalDirection)

        error_message = "Cannot create 1D profile. The 3D data was discarded and no matching 2D view was found."
        if energyMismatch:
            error_message += " Note that a view candidate was found to only differ in energy type."
        raise RuntimeError(error_message)

    def _extractHistogramFromView(self, view, horizontalDirection: Direction):
        if view.axisU == horizontalDirection.axis:
            axisToSum = 1
        else:
            axisToSum = 0

        data = view.getImageData(logScale=False, autoFlip=False)
        data = np.sum(data, axis=axisToSum)
        if axisToSum == 0:
            data = np.flip(data, axis=0)
        return data

    def _correctCapitalization(self, solidLabel, surfaceLabel):
        if solidLabel is None:
            return None, None
        originalSolidLabels = self._logger.getSeenSolidLabels()
        lowerCaseSolidLabels = [label.lower() for label in originalSolidLabels]
        if solidLabel.lower() in lowerCaseSolidLabels:
            labelIndex = lowerCaseSolidLabels.index(solidLabel.lower())
            solidLabel = originalSolidLabels[labelIndex]

        if surfaceLabel is None:
            return solidLabel, None

        originalSurfaceLabels = self._logger.getSeenSurfaceLabels(solidLabel)
        lowerCaseSurfaceLabels = [label.lower() for label in originalSurfaceLabels]

        altLabel = f"{solidLabel}_{surfaceLabel}"
        if altLabel.lower() in lowerCaseSurfaceLabels:
            surfaceLabel = altLabel

        if surfaceLabel.lower() in lowerCaseSurfaceLabels:
            labelIndex = lowerCaseSurfaceLabels.index(surfaceLabel.lower())
            surfaceLabel = originalSurfaceLabels[labelIndex]
        return solidLabel, surfaceLabel

    def _createName(
        self,
        horizontalDirection: Direction,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool,
        energyType: EnergyType,
    ) -> str:
        name = f"{energyType.name} profile along " + "xyz"[horizontalDirection.axis]
        if solidLabel:
            name += " of " + solidLabel
        if surfaceLabel:
            name += " surface " + surfaceLabel
            if surfaceEnergyLeaving:
                name += " (leaving)"
            else:
                name += " (entering)"
        return name
