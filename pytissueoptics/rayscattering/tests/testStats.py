import unittest

import numpy as np
from mockito import mock, when

from pytissueoptics.rayscattering import Stats
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.scene import Logger, Vector
from pytissueoptics.scene.logger import InteractionKey


class TestStats(unittest.TestCase):
    def setUp(self):
        source = mock(Source)
        when(source).getPhotonCount().thenReturn(1)
        self.stats = Stats(self.makeTestCubeLogger(),  source=source)

    def testWhenGet3DScatter_shouldReturnScatterOfAllSolidPoints(self):
        scatter = self.stats._get3DScatter()
        self.assertEqual(scatter.shape, (4, 8))
        self.assertTrue(np.array_equal(scatter[3], np.full(8, 0.1)))

    def testWhenGet3DScatterOfSurface_shouldReturnScatterOfAllPointsLeavingTheSurface(self):
        frontScatter = self.stats._get3DScatter(solidLabel="cube", surfaceLabel="front")
        self.assertEqual(0, len(frontScatter))

        backScatter = self.stats._get3DScatter(solidLabel="cube", surfaceLabel="back")
        self.assertEqual(backScatter.shape, (4, 1))
        self.assertEqual(0.2, backScatter[3])

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
        self.assertEqual([0.], x)
        self.assertEqual([1.], z)
        self.assertEqual([0.2], value)

    def testWhenGet1DScatterAlongZ_shouldReturnScatterOfAllSolidPointsProjectedOnZAxis(self):
        z, value = self.stats._get1DScatter(along="z")
        self.assertTrue(np.array_equal(value, np.full(8, 0.1)))
        self.assertTrue(len(z) == len(value))
        self.assertTrue(np.isclose(z, np.arange(0.1, 0.9, 0.1)).all())

    def testAbsorbanceOfSolid(self):
        self.assertAlmostEqual(0.8, self.stats.getAbsorbance("cube"))

    def testTransmittanceOfSolid(self):
        self.assertAlmostEqual(0.2, self.stats.getTransmittance("cube"))

    @staticmethod
    def makeTestCubeLogger() -> Logger:
        """ We log a few points taken from a unit cube centered at the origin where a single photon
        was propagated. We log one point entering front surface at z=0 with weight=1, then 8 points
        of weight 0.1 centered from z=0.1 to z=0.8, and one point exiting back surface at z=1 with
        a weight left of 0.2.
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
