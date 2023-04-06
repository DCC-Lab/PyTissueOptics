import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.display.views.defaultViews import View2DProjection
from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.rayscattering.opencl.CLScene import NO_SOLID_LABEL
from pytissueoptics.rayscattering.energyLogging import PointCloud, PointCloudFactory


@dataclass
class SurfaceStats:
    transmittance: float


@dataclass
class SolidStats:
    absorbance: float
    totalAbsorbance: float
    transmittance: float
    surfaces: Dict[str, SurfaceStats]


class Stats:
    def __init__(self, logger: EnergyLogger):
        self._logger = logger
        self._pointCloudFactory = PointCloudFactory(logger)
        self._extractFromViews = not logger.has3D

        try:
            self._photonCount = logger.info["photonCount"]
        except KeyError:
            raise RuntimeError("Logger is empty. Cannot compute statistics.")
        self._sourceSolidLabel = logger.info["sourceSolidLabel"]

        self._solidStatsMap = {}

    def report(self, solidLabel: str = None, saveToFile: str = None, verbose=True):
        if solidLabel and solidLabel not in self._logger.getSeenSolidLabels():
            utils.warn(f"WARNING: Cannot compute stats for solid '{solidLabel}' because it was not logged.")
            return

        self._computeStats(solidLabel)

        reportString = self._makeReport(solidLabel=solidLabel)
        if saveToFile:
            self._saveReport(reportString, saveToFile)
        if verbose:
            print(reportString)

    def _computeStats(self, solidLabel: str = None):
        solidLabels = [solidLabel]
        if solidLabel is None or utils.labelsEqual(solidLabel, NO_SOLID_LABEL):
            solidLabels = self._logger.getSeenSolidLabels()

        for solidLabel in solidLabels:
            if solidLabel == NO_SOLID_LABEL:
                continue
            try:
                absorbance = self.getAbsorbance(solidLabel)
            except ZeroDivisionError:
                utils.warn("WARNING: No energy input for solid '{}'".format(solidLabel))
                absorbance = None
            self._solidStatsMap[solidLabel] = SolidStats(absorbance,
                                                         self.getAbsorbance(solidLabel, useTotalEnergy=True),
                                                         self.getTransmittance(solidLabel),
                                                         self._getSurfaceStats(solidLabel))

    def _makeReport(self, solidLabel: str = None, reportString: str = ""):
        if solidLabel:
            if solidLabel == NO_SOLID_LABEL:
                reportString += self._reportWorld(solidLabel)
            else:
                reportString += self._reportSolid(solidLabel)
        else:
            for solidLabel in self._logger.getSeenSolidLabels():
                reportString = self._makeReport(solidLabel, reportString)
        return reportString

    def _reportWorld(self, worldLabel: str):
        totalSolidEnergy = sum([solidStats.totalAbsorbance for solidStats in self._solidStatsMap.values()])
        reportString = "Report of '{}'\n".format(worldLabel)
        reportString += "  Absorbed {:.2f}% of total power\n".format(100 - totalSolidEnergy)
        return reportString

    def _reportSolid(self, solidLabel: str):
        solidStats = self._solidStatsMap[solidLabel]
        reportString = "Report of solid '{}'\n".format(solidLabel)

        if solidStats.absorbance is None:
            reportString += ("  Absorbance: N/A ({:.2f}% of total power)\n".format(solidStats.totalAbsorbance))
            reportString += "  Absorbance + Transmittance: N/A\n"
            return reportString

        reportString += (
            "  Absorbance: {:.2f}% ({:.2f}% of total power)\n".format(solidStats.absorbance, solidStats.totalAbsorbance))
        reportString += ("  Absorbance + Transmittance: {:.1f}%\n".format(solidStats.absorbance + solidStats.transmittance))

        for surfaceLabel, surfaceStats in solidStats.surfaces.items():
            reportString += "    Transmittance at '{}': {:.1f}%\n".format(surfaceLabel, surfaceStats.transmittance)

        return reportString

    def getAbsorbance(self, solidLabel: str, useTotalEnergy=False) -> float:
        if self._extractFromViews:
            return self._getAbsorbanceFromViews(solidLabel, useTotalEnergy)
        points = self._getPointCloud(solidLabel).solidPoints
        energyInput = self.getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return 100 * self._sumEnergy(points) / energyInput

    def _getAbsorbanceFromViews(self, solidLabel: str, useTotalEnergy=False) -> float:
        energyInput = self.getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        absorbedEnergy = self._getAbsorbedEnergyFromViews(solidLabel)
        return 100 * absorbedEnergy / energyInput

    def _getAbsorbedEnergyFromViews(self, solidLabel: str) -> float:
        for view in self._logger.views:
            if not isinstance(view, View2DProjection):
                continue
            if not utils.labelsEqual(solidLabel, view.solidLabel):
                continue
            if not self._viewContainsSolid(view, solidLabel):
                continue
            return view.getSum()

        raise Exception(f"Could not extract absorbance for solid '{solidLabel}'. The 3D data was discarded and "
                        f"no stored 2D view corresponds to this solid.")

    def _viewContainsSolid(self, view, solidLabel: str) -> bool:
        solidLimits = self._logger.getSolidLimits(solidLabel)
        requiredLimitsU = solidLimits[view.axisU]
        requiredLimitsV = solidLimits[view.axisV]
        if min(view.limitsU) > min(requiredLimitsU) or max(view.limitsU) < max(requiredLimitsU):
            return False
        if min(view.limitsV) > min(requiredLimitsV) or max(view.limitsV) < max(requiredLimitsV):
            return False
        return True

    def getPhotonCount(self) -> int:
        return self._photonCount

    def getEnergyInput(self, solidLabel: str = None) -> float:
        if solidLabel is None:
            return self.getPhotonCount()
        if self._extractFromViews:
            return self._getEnergyInputFromViews(solidLabel)
        points = self._getPointCloudOfSurfaces(solidLabel).enteringSurfacePoints
        energy = self._sumEnergy(points)

        if utils.labelsEqual(self._sourceSolidLabel, solidLabel):
            energy += self.getPhotonCount()
        return energy

    def _getEnergyInputFromViews(self, solidLabel: str) -> float:
        return self._getEnergyCrossingSolidFromViews(solidLabel, leaving=False)

    def _getEnergyLeavingFromViews(self, solidLabel: str):
        return self._getEnergyCrossingSolidFromViews(solidLabel, leaving=True)

    def _getEnergyCrossingSolidFromViews(self, solidLabel: str, leaving: bool) -> float:
        energy = 0
        for surfaceLabel in self._logger.getSeenSurfaceLabels(solidLabel):
            energy += self._getSurfaceEnergyFromViews(solidLabel, surfaceLabel, leaving=leaving)

        if utils.labelsEqual(self._sourceSolidLabel, solidLabel):
            energy += self.getPhotonCount()
        return energy

    def _getSurfaceEnergyFromViews(self, solidLabel: str, surfaceLabel: str, leaving: bool) -> float:
        for view in self._logger.views:
            if not utils.labelsEqual(solidLabel, view.solidLabel):
                continue
            if not utils.labelsEqual(surfaceLabel, view.surfaceLabel):
                continue
            if view.surfaceEnergyLeaving != leaving:
                continue
            if not self._viewContainsSolid(view, solidLabel):
                continue
            return view.getSum()
        raise Exception(f"Could not extract energy {['entering', 'leaving'][leaving]} surface '{surfaceLabel}' "
                        f"of solid '{solidLabel}'. The 3D data was discarded and no stored 2D view corresponds "
                        f"to this surface.")

    def _getSurfaceStats(self, solidLabel: str) -> Dict[str, SurfaceStats]:
        stats = {}
        for surfaceLabel in self._logger.getSeenSurfaceLabels(solidLabel):
            stats[surfaceLabel] = SurfaceStats(self.getTransmittance(solidLabel, surfaceLabel))
        return stats

    def getTransmittance(self, solidLabel: str, surfaceLabel: str = None, useTotalEnergy=False):
        """ Uses local energy input for the desired solid by default. Specify 'useTotalEnergy' = True
        to compare instead with total input energy of the scene. """
        if self._extractFromViews:
            return self._getTransmittanceFromViews(solidLabel, surfaceLabel, useTotalEnergy)

        if surfaceLabel is None:
            points = self._getPointCloudOfSurfaces(solidLabel).leavingSurfacePoints
        else:
            points = self._getPointCloud(solidLabel, surfaceLabel).leavingSurfacePoints

        energyInput = self.getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return 100 * self._sumEnergy(points) / energyInput

    def _getTransmittanceFromViews(self, solidLabel: str, surfaceLabel: str = None, useTotalEnergy=False):
        if surfaceLabel is None:
            energyLeaving = self._getEnergyLeavingFromViews(solidLabel)
        else:
            energyLeaving = self._getSurfaceEnergyFromViews(solidLabel, surfaceLabel, leaving=True)

        energyInput = self.getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return 100 * energyLeaving / energyInput

    @staticmethod
    def _sumEnergy(points: np.ndarray):
        return np.abs(np.sum(points[:, 0])) if points is not None else 0

    def _getPointCloud(self, solidLabel: str = None, surfaceLabel: str = None) -> PointCloud:
        return self._pointCloudFactory.getPointCloud(solidLabel, surfaceLabel)

    def _getPointCloudOfSurfaces(self, solidLabel: str = None) -> PointCloud:
        return self._pointCloudFactory.getPointCloudOfSurfaces(solidLabel)

    @staticmethod
    def _saveReport(report: str, filepath: str = None):
        if filepath is None:
            filepath = "simulation_report"
            utils.warn(f"WARNING: No filepath specified. Saving to {filepath}.")
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
