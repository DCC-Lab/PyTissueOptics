import unittest

import numpy as np
from mockito import mock, when

from rayscattering.statistics.statistics import Stats
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.scene import Logger, Vector
from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.logger import InteractionKey
from pytissueoptics.scene.solids import Solid


@unittest.skip("Deprecated.")
class TestStats(unittest.TestCase):
    def setUp(self):
        source = self.makeTestSource()
        self.stats = Stats(self.makeTestCubeLogger(),  source=source)

    def testWhenGet3DScatter_shouldReturnScatterOfAllSolidPoints(self):
        scatter = self.stats._get3DScatter()
        self.assertEqual(scatter.shape, (8, 4))
        self.assertTrue(np.array_equal(scatter[:, 3], np.full(8, 0.1)))

    def testGivenEmptyLogger_whenGet3DScatter_shouldReturnEmptyArray(self):
        logger = Logger()
        stats = Stats(logger, self.makeTestSource())

        scatter = stats._get3DScatter()

        self.assertTrue(len(scatter) == 0)

    def testWhenGet3DScatterOfSurface_shouldReturnScatterOfAllPointsLeavingTheSurface(self):
        frontScatter = self.stats._get3DScatter(solidLabel="cube", surfaceLabel="front")
        self.assertEqual(0, frontScatter.size)

        backScatter = self.stats._get3DScatter(solidLabel="cube", surfaceLabel="back")
        self.assertEqual(backScatter.shape, (1, 4))
        self.assertEqual(0.2, backScatter[:, 3])

    def testWhenGet3DScatterOfPointsEnteringSurface_shouldReturnScatterOfAllPointsEnteringTheSurface(self):
        frontScatter = self.stats._get3DScatter(solidLabel="cube", surfaceLabel="front", enteringSurface=True)
        self.assertEqual(frontScatter.shape, (1, 4))
        self.assertEqual(1, frontScatter[:, 3])

        backScatter = self.stats._get3DScatter(solidLabel="cube", surfaceLabel="back", enteringSurface=True)
        self.assertEqual(0, backScatter.size)

    def testWhenGet2DScatterWithYProjection_shouldReturnScatterOfAllSolidPointsProjectedToXZ(self):
        x, z, value = self.stats._get2DScatter(projection="y")
        self.assertTrue(np.array_equal(value, np.full(8, 0.1)))
        self.assertTrue(len(x) == len(z) == len(value))
        self.assertTrue(np.array_equal(x, np.zeros(8)))
        self.assertTrue(np.isclose(z, np.arange(0.1, 0.9, 0.1)).all())

    def testWhenGet2DScatterOfSurface_shouldReturnScatterOfAllPointsLeavingTheSurface(self):
        frontScatter = self.stats._get2DScatter(solidLabel="cube", surfaceLabel="front")
        x, z, value = frontScatter
        self.assertEqual(0, len(value))

        backScatter = self.stats._get2DScatter(solidLabel="cube", surfaceLabel="back")
        x, z, value = backScatter
        self.assertEqual(0, x)
        self.assertEqual(1, z)
        self.assertEqual(0.2, value)

    def testWhenGet1DScatterAlongZ_shouldReturnScatterOfAllSolidPointsProjectedOnZAxis(self):
        z, value = self.stats._get1DScatter(along="z")
        self.assertTrue(np.array_equal(value, np.full(8, 0.1)))
        self.assertTrue(len(z) == len(value))
        self.assertTrue(np.isclose(z, np.arange(0.1, 0.9, 0.1)).all())

    def testWhenGetEnergyInput_shouldReturnSumOfAllEnergyEnteringTheSolidSurfaces(self):
        energyInput = self.stats._getEnergyInput("cube")
        self.assertEqual(1, energyInput)

    def testGivenSourceInSolid_whenGetEnergyInput_shouldAddSourceEnergyToSolidInputEnergy(self):
        solid = mock(Solid)
        when(solid).getLabel().thenReturn("cube")
        source = self.makeTestSource(solid)
        self.stats = Stats(self.makeTestCubeLogger(), source=source)

        energy = self.stats._getEnergyInput("cube")

        self.assertEqual(1 + 1, energy)

    def testAbsorbanceOfSolid(self):
        self.assertAlmostEqual(0.8, self.stats.getAbsorbance("cube"))

    def testTransmittanceOfSolid(self):
        self.assertAlmostEqual(0.2, self.stats.getTransmittance("cube"))

    def testGivenALoggerWithNoSolidInteractions_whenGet3DScatter_shouldWarnAndReturnEmptyArray(self):
        logger = self.makeTestCubeLogger()
        logger._data.pop(InteractionKey("cube"))
        stats = Stats(logger, source=self.makeTestSource())

        with self.assertWarns(UserWarning):
            self.assertEqual(0, len(stats._get3DScatter(solidLabel="cube")))

    @staticmethod
    def makeTestCubeLogger() -> Logger:
        """ We log a few points taken from a unit cube centered at the origin where a single photon
        was propagated. We log one point entering front surface at z=0 with weight=1, then 8 points
        of weight 0.1 centered from z=0.1 to z=0.8, and one point exiting back surface at z=1 with
        a remaining weight of 0.2 so it correctly adds up to 1.
        """
        logger = Logger()
        solidInteraction = InteractionKey("cube")
        frontInteraction = InteractionKey("cube", "front")
        backInteraction = InteractionKey("cube", "back")
        for i in range(1, 9):
            logger.logDataPoint(0.1, Vector(0, 0, 0.1*i), solidInteraction)
        logger.logDataPoint(-1, Vector(0, 0, 0), frontInteraction)
        logger.logDataPoint(0.2, Vector(0, 0, 1), backInteraction)
        return logger

    @staticmethod
    def makeTestSource(inSolid: Solid = None):
        source = mock(Source)
        sourceEnv = Environment(None, inSolid)
        when(source).getPhotonCount().thenReturn(1)
        when(source).getEnvironment().thenReturn(sourceEnv)
        return source
