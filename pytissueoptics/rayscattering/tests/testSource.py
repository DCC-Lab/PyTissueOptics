import os
import tempfile
import unittest

import numpy as np
from mockito import mock, when, verify

from pytissueoptics.rayscattering import PencilPointSource, Photon, EnergyLogger
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.source import Source, IsotropicPointSource, DirectionalSource, DivergentSource
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.scene.geometry import Environment, Vector
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.logger import Logger


class TestSource(unittest.TestCase):
    SOURCE_ENV = Environment(ScatteringMaterial())
    SOURCE_POSITION = Vector(0, 0, 0)

    def setUp(self):
        self.photon = self._createPhoton()
        self.source = SinglePhotonSource(position=Vector(), photons=[self.photon])

    def testWhenPropagate_shouldSetInitialPhotonEnvironmentAsSourceEnvironment(self):
        self.source.propagate(self._createTissue(), showProgress=False)
        verify(self.photon).setContext(self.SOURCE_ENV, ...)

    def testWhenPropagate_shouldPropagateAllPhotons(self):
        self.source.propagate(self._createTissue(), showProgress=False)
        verify(self.photon).propagate()

    def testWhenPropagate_shouldUpdatePhotonCountInLogger(self):
        logger = EnergyLogger(mock(ScatteringScene), views=[])
        self.source.propagate(self._createTissue(), logger=logger, showProgress=False)
        self.assertEqual(logger.info['photonCount'], 1)

        logger.info['photonCount'] = 10
        self.source.propagate(self._createTissue(), logger=logger, showProgress=False)
        self.assertEqual(logger.info['photonCount'], 10+1)

    def testWhenPropagate_shouldSetSourceSolidLabelInLogger(self):
        sourceSolidLabel = 'the source solid'
        solid = mock(Solid)
        when(solid).getLabel().thenReturn(sourceSolidLabel)
        self.SOURCE_ENV = Environment(ScatteringMaterial(), solid)
        logger = EnergyLogger(mock(ScatteringScene), views=[])

        self.source.propagate(self._createTissue(), logger=logger, showProgress=False)

        self.assertEqual(logger.info['sourceSolidLabel'], sourceSolidLabel)

    def testGivenSourceNotInASolid_whenPropagate_shouldSetSourceSolidLabelInLoggerAsNone(self):
        logger = EnergyLogger(mock(ScatteringScene), views=[])
        self.source.propagate(self._createTissue(), logger=logger, showProgress=False)
        self.assertEqual(logger.info['sourceSolidLabel'], None)

    def testWhenPropagate_shouldSetSourceHashInLogger(self):
        logger = EnergyLogger(mock(ScatteringScene), views=[])
        self.source.propagate(self._createTissue(), logger=logger, showProgress=False)
        self.assertEqual(logger.info['sourceHash'], hash(self.source))

    def testGivenLoggerWithFilePath_whenPropagate_shouldSaveLogger(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, 'test.log')
            with self.assertWarns(UserWarning):
                logger = EnergyLogger(mock(ScatteringScene), views=[], filepath=filepath)

            self.source.propagate(self._createTissue(), logger=logger, showProgress=False)

            self.assertTrue(os.path.exists(filepath))

    def testGivenLoggerUsedOnADifferentSource_whenPropagate_shouldWarn(self):
        logger = EnergyLogger(mock(ScatteringScene), views=[])
        logger.info['sourceHash'] = 1234
        with self.assertWarns(UserWarning):
            self.source.propagate(self._createTissue(), logger=logger, showProgress=False)

    def testGivenAnInstanceOfBaseLogger_whenPropagate_shouldWarn(self):
        logger = Logger()
        with self.assertWarns(UserWarning):
            self.source.propagate(self._createTissue(), logger=logger, showProgress=False)

    def _createTissue(self):
        tissue = mock(ScatteringScene)
        when(tissue).getEnvironmentAt(self.SOURCE_POSITION).thenReturn(self.SOURCE_ENV)
        when(tissue).resetOutsideMaterial(...).thenReturn()
        when(tissue).getBoundingBox().thenReturn()
        when(tissue).getPolygons().thenReturn([])
        when(tissue).getSolids().thenReturn([])
        return tissue

    @staticmethod
    def _createPhoton():
        photon = mock(Photon)
        when(photon).setContext(...).thenReturn()
        when(photon).propagate(...).thenReturn()
        return photon


class SinglePhotonSource(Source):
    def __init__(self, position, photons):
        super().__init__(position, N=len(photons), useHardwareAcceleration=False)
        self._photons = photons

    def getInitialPositionsAndDirections(self):
        return np.array([[0, 0, 0]]), np.array([[0, 0, 1]])

    @property
    def _hashComponents(self) -> tuple:
        return self._position,


class TestPencilSource(unittest.TestCase):
    def testShouldHavePhotonsAllPointingInTheSourceDirection(self):
        sourceDirection = Vector(1, 0, 0)
        pencilSource = PencilPointSource(position=Vector(), direction=sourceDirection, N=10)
        for photon in pencilSource.photons:
            self.assertEqual(sourceDirection, photon.direction)

    def testShouldHavePhotonsAllPositionedAtTheSourcePosition(self):
        sourcePosition = Vector(3, 3, 0)
        pencilSource = PencilPointSource(position=sourcePosition, direction=Vector(0, 0, 1), N=10)
        for photon in pencilSource.photons:
            self.assertEqual(sourcePosition, photon.position)


class TestIsotropicPointSource(unittest.TestCase):
    def testShouldHavePhotonsAllPositionedAtTheSourcePosition(self):
        sourcePosition = Vector(3, 3, 0)
        pointSource = IsotropicPointSource(position=sourcePosition, N=10, useHardwareAcceleration=False)
        for photon in pointSource.photons:
            self.assertEqual(sourcePosition, photon.position)

    def testGivenTwoIsotropicSourcesWithSamePropertiesExceptPhotonCount_shouldHaveSameHash(self):
        source1 = IsotropicPointSource(position=Vector(), N=1, useHardwareAcceleration=False)
        source2 = IsotropicPointSource(position=Vector(), N=2, useHardwareAcceleration=False)
        self.assertEqual(hash(source1), hash(source2))

    def testGivenTwoIsotropicSourcesThatDifferInPosition_shouldNotHaveSameHash(self):
        source1 = IsotropicPointSource(position=Vector(), N=1, useHardwareAcceleration=False)
        source2 = IsotropicPointSource(position=Vector(1, 0, 0), N=1, useHardwareAcceleration=False)
        self.assertNotEqual(hash(source1), hash(source2))


class TestDirectionalSource(unittest.TestCase):
    def testShouldHavePhotonsAllPointingInTheSourceDirection(self):
        sourceDirection = Vector(1, 0, 0)
        directionalSource = DirectionalSource(position=Vector(), direction=sourceDirection, diameter=1, N=10)
        for photon in directionalSource.photons:
            self.assertEqual(sourceDirection, photon.direction)

    def testShouldHavePhotonsUniformlyPositionedInsideTheSourceDiameter(self):
        np.random.seed(0)
        sourcePosition = Vector(3, 3, 0)
        sourceDiameter = 2
        directionalSourceTowardsY = DirectionalSource(position=sourcePosition, direction=Vector(0, 1, 0),
                                                      diameter=sourceDiameter, N=10)
        for photon in directionalSourceTowardsY.photons:
            self.assertTrue(np.isclose(photon.position.y, sourcePosition.y))
            self.assertTrue(photon.position.x <= sourcePosition.x + sourceDiameter / 2)
            self.assertTrue(photon.position.x >= sourcePosition.x - sourceDiameter / 2)
            self.assertTrue(photon.position.z <= sourcePosition.z + sourceDiameter / 2)
            self.assertTrue(photon.position.z >= sourcePosition.z - sourceDiameter / 2)

        self.assertFalse(all(photon.position.x == sourcePosition.x for photon in directionalSourceTowardsY.photons))
        self.assertFalse(all(photon.position.z == sourcePosition.z for photon in directionalSourceTowardsY.photons))

    def testGivenTwoDirectionalSourcesWithSamePropertiesExceptPhotonCount_shouldHaveSameHash(self):
        source1 = DirectionalSource(position=Vector(), direction=Vector(1, 0, 0), diameter=1, N=1)
        source2 = DirectionalSource(position=Vector(), direction=Vector(1, 0, 0), diameter=1, N=2)
        self.assertEqual(hash(source1), hash(source2))

    def testGivenTwoDirectionalSourcesThatDifferInDiameter_shouldHaveDifferentHash(self):
        source1 = DirectionalSource(position=Vector(), direction=Vector(1, 0, 0), diameter=1, N=1)
        source2 = DirectionalSource(position=Vector(), direction=Vector(1, 0, 0), diameter=2, N=1)
        self.assertNotEqual(hash(source1), hash(source2))


class TestDivergentSource(unittest.TestCase):
    def testShouldHavePhotonsUniformlyPositionedInsideTheSourceDiameter(self):
        np.random.seed(0)
        sourcePosition = Vector(3, 3, 0)
        sourceDiameter = 2
        divergentSourceTowardsY = DivergentSource(position=sourcePosition, direction=Vector(0, 1, 0),
                                                    diameter=sourceDiameter, divergence=0.2, N=10)
        for photon in divergentSourceTowardsY.photons:
            self.assertTrue(np.isclose(photon.position.y, sourcePosition.y))
            self.assertTrue(photon.position.x <= sourcePosition.x + sourceDiameter / 2)
            self.assertTrue(photon.position.x >= sourcePosition.x - sourceDiameter / 2)
            self.assertTrue(photon.position.z <= sourcePosition.z + sourceDiameter / 2)
            self.assertTrue(photon.position.z >= sourcePosition.z - sourceDiameter / 2)

        self.assertFalse(all(photon.position.x == sourcePosition.x for photon in divergentSourceTowardsY.photons))
        self.assertFalse(all(photon.position.z == sourcePosition.z for photon in divergentSourceTowardsY.photons))

    def testGivenNoDivergence_shouldHavePhotonsAllPointingInTheSourceDirection(self):
        sourceDirection = Vector(1, 0, 0)
        divergentSource = DivergentSource(position=Vector(), direction=sourceDirection, diameter=1, divergence=0, N=10)
        for photon in divergentSource.photons:
            self.assertEqual(sourceDirection, photon.direction)

    def testGivenDivergence_shouldHavePhotonsPointingInDifferentDirectionsAroundTheSourceDirection(self):
        sourceDirection = Vector(1, 0, 0)
        divergence = np.pi/4
        divergentSource = DivergentSource(position=Vector(), direction=sourceDirection, diameter=1,
                                          divergence=divergence, N=10)
        minDot = np.cos(divergence/2)
        for photon in divergentSource.photons:
            dot = photon.direction.dot(sourceDirection)
            self.assertTrue(dot >= minDot)

    def testGivenTwoDivergentSourcesWithSamePropertiesExceptPhotonCount_shouldHaveSameHash(self):
        sourceDirection = Vector(1, 0, 0)
        divergence = np.pi/4
        divergentSource1 = DivergentSource(position=Vector(), direction=sourceDirection, diameter=1,
                                           divergence=divergence, N=1)
        divergentSource2 = DivergentSource(position=Vector(), direction=sourceDirection, diameter=1,
                                           divergence=divergence, N=2)
        self.assertEqual(hash(divergentSource1), hash(divergentSource2))

    def testGivenTwoDivergentSourcesThatDifferInDivergence_shouldNotHaveSameHash(self):
        sourceDirection = Vector(1, 0, 0)
        divergence1 = np.pi/4
        divergence2 = np.pi/2
        divergentSource1 = DivergentSource(position=Vector(), direction=sourceDirection, diameter=1,
                                           divergence=divergence1, N=1)
        divergentSource2 = DivergentSource(position=Vector(), direction=sourceDirection, diameter=1,
                                           divergence=divergence2, N=1)
        self.assertNotEqual(hash(divergentSource1), hash(divergentSource2))
