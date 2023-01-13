import os

import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.energyLogger import EnergyLogger
from pytissueoptics.rayscattering.pointCloud import PointCloudFactory, PointCloud
from pytissueoptics.rayscattering.opencl import warnings


class Stats:
    def __init__(self, logger: EnergyLogger):
        self._logger = logger
        self._pointCloudFactory = PointCloudFactory(logger)
        self._extractFromViews = not logger.has3D

        self._photonCount = logger.info["photonCount"]
        self._sourceSolidLabel = logger.info["sourceSolidLabel"]

    def report(self, solidLabel: str = None, saveToFile: str = None, verbose=True):
        reportString = self._makeReport(solidLabel=solidLabel)
        if saveToFile:
            self.saveReport(reportString, saveToFile)
        if verbose:
            print(reportString)

    def _makeReport(self, solidLabel: str = None, reportString: str = ""):
        if solidLabel:
            reportString += self._reportSolid(solidLabel)
        else:
            for solidLabel in self._logger.getSolidLabels():
                reportString = self._makeReport(solidLabel, reportString)
        return reportString

    def _reportSolid(self, solidLabel: str):
        reportString = "Report of solid '{}'\n".format(solidLabel)
        try:
            reportString += (
                "  Absorbance: {:.2f}% ({:.2f}% of total power)\n".format(100 * self.getAbsorbance(solidLabel),
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

    def getAbsorbance(self, solidLabel: str, useTotalEnergy=False) -> float:
        if self._extractFromViews:
            return self._getAbsorbanceFromViews(solidLabel, useTotalEnergy)
        points = self._getPointCloud(solidLabel).solidPoints
        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return self._sumEnergy(points) / energyInput

    def _getAbsorbanceFromViews(self, solidLabel: str, useTotalEnergy=False) -> float:
        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        absorbedEnergy = self._getAbsorbedEnergyFromViews(solidLabel)
        return absorbedEnergy / energyInput

    def _getAbsorbedEnergyFromViews(self, solidLabel: str) -> float:
        for view in self._logger.views:
            if not utils.labelsEqual(solidLabel, view.solidLabel):
                continue
            if view.surfaceLabel is not None:
                continue
            # todo: make sure the limits correspond to the solid limits
            return view.getSum()

        raise Exception(f"Could not extract absorbance for solid '{solidLabel}'. The 3D data was discarded and "
                        f"no stored 2D view corresponds to this solid.")

    def getPhotonCount(self) -> int:
        return self._photonCount

    def _getEnergyInput(self, solidLabel: str = None) -> float:
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
        for surfaceLabel in self._logger.getSurfaceLabels(solidLabel):
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
            return view.getSum()
        raise Exception(f"Could not extract energy {['entering', 'leaving'][leaving]} surface '{surfaceLabel}' "
                        f"of solid '{solidLabel}'. The 3D data was discarded and no stored 2D view corresponds "
                        f"to this surface.")

    def getTransmittance(self, solidLabel: str, surfaceLabel: str = None, useTotalEnergy=False):
        """ Uses local energy input for the desired solid by default. Specify 'useTotalEnergy' = True
        to compare instead with total input energy of the scene. """
        if self._extractFromViews:
            return self._getTransmittanceFromViews(solidLabel, surfaceLabel, useTotalEnergy)

        if surfaceLabel is None:
            points = self._getPointCloudOfSurfaces(solidLabel).leavingSurfacePoints
        else:
            points = self._getPointCloud(solidLabel, surfaceLabel).leavingSurfacePoints

        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return self._sumEnergy(points) / energyInput

    def _getTransmittanceFromViews(self, solidLabel: str, surfaceLabel: str = None, useTotalEnergy=False):
        if surfaceLabel is None:
            energyLeaving = self._getEnergyLeavingFromViews(solidLabel)
        else:
            energyLeaving = self._getSurfaceEnergyFromViews(solidLabel, surfaceLabel, leaving=True)

        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return energyLeaving / energyInput

    @staticmethod
    def _sumEnergy(points: np.ndarray):
        return np.abs(np.sum(points[:, 0])) if points is not None else 0

    def _getPointCloud(self, solidLabel: str = None, surfaceLabel: str = None) -> PointCloud:
        return self._pointCloudFactory.getPointCloud(solidLabel, surfaceLabel)

    def _getPointCloudOfSurfaces(self, solidLabel: str = None) -> PointCloud:
        return self._pointCloudFactory.getPointCloudOfSurfaces(solidLabel)

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
