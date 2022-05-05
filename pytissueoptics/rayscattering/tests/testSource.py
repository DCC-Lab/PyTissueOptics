import unittest

import numpy as np
from mockito import mock, when, verify

from pytissueoptics.rayscattering import PencilSource, Photon
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.source import Source, IsotropicPointSource
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger
from pytissueoptics.scene.geometry import Environment, Vector


class TestSource(unittest.TestCase):
    SOURCE_ENV = Environment(ScatteringMaterial())
    SOURCE_POSITION = Vector(0, 0, 0)

    def setUp(self):
        self.photon = self._createPhoton()
        self.source = SinglePhotonSource(position=Vector(), photons=[self.photon])

    def testWhenPropagate_shouldSetInitialPhotonEnvironmentAsSourceEnvironment(self):
        self.source.propagate(self._createTissue(), progressBar=False)
        verify(self.photon).setContext(self.SOURCE_ENV, ...)

    def testWhenPropagate_shouldUpdatePhotonCountInLogger(self):
        logger = Logger()
        self.source.propagate(self._createTissue(), logger=logger, progressBar=False)
        self.assertEqual(logger.info['photonCount'], 1)

        logger.info['photonCount'] = 10
        self.source.propagate(self._createTissue(), logger=logger, progressBar=False)
        self.assertEqual(logger.info['photonCount'], 10+1)

    def testWhenPropagate_shouldPropagateAllPhotons(self):
        self.source.propagate(self._createTissue(), progressBar=False)
        verify(self.photon).propagate()

    def _createTissue(self):
        tissue = mock(RayScatteringScene)
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


class TestPencilSource(unittest.TestCase):
    def testShouldHavePhotonsAllPointingInTheSourceDirection(self):
        sourceDirection = Vector(1, 0, 0)
        pencilSource = PencilSource(position=Vector(), direction=sourceDirection, N=10)
        for photon in pencilSource.photons:
            self.assertEqual(sourceDirection, photon.direction)

    def testShouldHavePhotonsAllPositionedAtTheSourcePosition(self):
        sourcePosition = Vector(3, 3, 0)
        pencilSource = PencilSource(position=sourcePosition, direction=Vector(0, 0, 1), N=10)
        for photon in pencilSource.photons:
            self.assertEqual(sourcePosition, photon.position)


class TestIsotropicPointSource(unittest.TestCase):
    def testShouldHavePhotonsAllPositionedAtTheSourcePosition(self):
        sourcePosition = Vector(3, 3, 0)
        pointSource = IsotropicPointSource(position=sourcePosition, N=10, useHardwareAcceleration=False)
        for photon in pointSource.photons:
            self.assertEqual(sourcePosition, photon.position)


class SinglePhotonSource(Source):
    def __init__(self, position, photons):
        super().__init__(position, N=len(photons), useHardwareAcceleration=False)
        self._photons = photons

    def getInitialPositionsAndDirections(self):
        return np.array([[0, 0, 0]]), np.array([[0, 0, 1]])
