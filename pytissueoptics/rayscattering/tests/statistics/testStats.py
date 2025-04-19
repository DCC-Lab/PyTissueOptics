import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.statistics import Stats
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger import InteractionKey
from pytissueoptics.scene.solids import Cube


class TestStats(unittest.TestCase):
    EXPECTED_SOLID_REPORT_LINES = ["Report of solid 'cube'",
                                   "  Absorbance: 80.00% (80.00% of total power)",
                                   "  Absorbance + Transmittance: 100.0%",
                                   "    Transmittance at 'cube_front': 0.0%",
                                   "    Transmittance at 'cube_back': 20.0%",
                                   '']
    EXPECTED_REPORT_LINES = EXPECTED_SOLID_REPORT_LINES[:-1] + ["Report of 'world'",
                                                                "  Absorbed 20.00% of total power",
                                                                '']

    def _setUp(self, keep3D=True, sourceSolidLabel=None, noViews=False):
        logger = self.makeTestCubeLogger(keep3D=keep3D, sourceSolidLabel=sourceSolidLabel, noViews=noViews)
        self.stats = Stats(logger)

    def testWhenGetEnergyInput_shouldReturnTotalPhotonCount(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)
                energy = self.stats.getEnergyInput()
                self.assertEqual(1, energy)

    def testWhenGetEnergyInputOfSolid_shouldReturnSumOfAllEnergyEnteringTheSolidSurfaces(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)
                energy = self.stats.getEnergyInput("cube")
                self.assertEqual(1, energy)

    def testGiven2DLoggerWithNoViewsOfSolid_whenGetAbsorbanceOfSolid_shouldRaiseException(self):
        self._setUp(keep3D=False, noViews=True)
        with self.assertRaises(Exception):
            self.stats.getAbsorbance("cube")

    def testGiven2DLoggerWithNoViewsOfSolid_whenGetEnergyInputOfSolid_shouldRaiseException(self):
        self._setUp(keep3D=False, noViews=True)
        with self.assertRaises(Exception):
            self.stats.getEnergyInput("cube")

    def testGiven2DLoggerWithNoViewsOfSurface_whenGetTransmittanceOfSurface_shouldRaiseException(self):
        self._setUp(keep3D=False, noViews=True)
        with self.assertRaises(Exception):
            self.stats.getTransmittance("cube", "cube_front")

    def testGivenSourceInSolid_whenGetEnergyInput_shouldAddSourceEnergyToSolidInputEnergy(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D, sourceSolidLabel="cube")
                energy = self.stats.getEnergyInput("cube")
                self.assertEqual(1 + 1, energy)

    def testWhenGetAbsorbanceOfSolid_shouldReturnAbsorbanceInPercentage(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)
                self.assertAlmostEqual(80, self.stats.getAbsorbance("cube"), places=5)

    def testWhenGetTransmittanceOfSolid_shouldReturnTransmittanceOfWholeSolid(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)
                self.assertAlmostEqual(20, self.stats.getTransmittance("cube"), places=5)

    def testWhenGetTransmittanceOfSolidSurface_shouldReturnTransmittanceAtThisSurface(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)
                self.assertAlmostEqual(0, self.stats.getTransmittance("cube", "cube_front"), places=5)
                self.assertAlmostEqual(20, self.stats.getTransmittance("cube", "cube_back"), places=5)

    def testWhenReportSolid_shouldPrintAFullReportOfThisSolidAndItsSurfaces(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)

                with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                    self.stats.report(solidLabel="cube")
                    reportLines = mock_stdout.getvalue().splitlines()

                self.assertEqual(self.EXPECTED_SOLID_REPORT_LINES, reportLines)

    def testWhenReport_shouldPrintAFullReportOfAllSolidsAndTheirSurfaces(self):
        for keep3D in [False, True]:
            with self.subTest(["using2DLogger", "using3DLogger"][keep3D]):
                self._setUp(keep3D=keep3D)

                with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                    self.stats.report()
                    reportLines = mock_stdout.getvalue().splitlines()

                self.assertEqual(self.EXPECTED_REPORT_LINES, reportLines)

    def testWhenReportToFile_shouldWriteReportToFile(self):
        with tempfile.TemporaryDirectory() as tempDir:
            filename = os.path.join(tempDir, "report.txt")
            self._setUp()

            self.stats.report(saveToFile=filename, verbose=False)

            with open(filename, "r") as file:
                reportLines = file.read().splitlines()

            self.assertEqual(self.EXPECTED_REPORT_LINES[:-1], reportLines)

    def testWhenReportNonExistingSolid_shouldWarnAndIgnore(self):
        self._setUp(keep3D=True)
        with self.assertWarns(UserWarning):
            self.stats.report(solidLabel="non-existing")

    @staticmethod
    def makeTestCubeLogger(keep3D=True, sourceSolidLabel=None, noViews=False) -> EnergyLogger:
        """ We log a few points taken from a unit cube centered at the origin where a single photon
        was propagated. We log one point entering front surface at z=0 with weight=1, then 8 points
        of weight 0.1 centered from z=0.1 to z=0.8, and one point exiting back surface at z=1 with
        a remaining weight of 0.2, so it correctly adds up to 1.
        """
        cube = Cube(1, position=Vector(0, 0, 0.5), material=ScatteringMaterial())
        scene = ScatteringScene([cube])
        logger = EnergyLogger(scene, keep3D=keep3D)
        if noViews:
            logger = EnergyLogger(scene, keep3D=keep3D, views=[])
        solidInteraction = InteractionKey("cube")
        frontInteraction = InteractionKey("cube", "cube_front")
        backInteraction = InteractionKey("cube", "cube_back")
        for i in range(1, 9):
            logger.logDataPoint(0.1, Vector(0, 0, 0.1*i), solidInteraction)
        logger.logDataPoint(-1, Vector(0, 0, 0), frontInteraction)
        logger.logDataPoint(0.2, Vector(0, 0, 1), backInteraction)
        logger.info["photonCount"] = 1
        logger.info["sourceSolidLabel"] = sourceSolidLabel

        # Logging the energy that left the solid on the outside world
        logger.logDataPoint(0.2, Vector(0, 0, 10), InteractionKey("world"))
        return logger
