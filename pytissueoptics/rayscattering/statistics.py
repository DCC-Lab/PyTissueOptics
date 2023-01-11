import os

import numpy as np

from pytissueoptics.rayscattering.pointCloud import PointCloudFactory, PointCloud
from pytissueoptics.rayscattering.opencl import warnings
from pytissueoptics.scene.logger import Logger


class Stats:
    def __init__(self, logger: Logger):
        self._logger = logger
        self._pointCloudFactory = PointCloudFactory(logger)

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

    def getAbsorbance(self, solidLabel: str = None, useTotalEnergy=False) -> float:
        points = self._getPointCloud(solidLabel).solidPoints
        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return self._sumEnergy(points) / energyInput

    def getPhotonCount(self) -> int:
        return self._photonCount

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
            points = self._getPointCloud(solidLabel, surfaceLabel).leavingSurfacePoints

        energyInput = self._getEnergyInput(solidLabel) if not useTotalEnergy else self.getPhotonCount()
        return self._sumEnergy(points) / energyInput

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
