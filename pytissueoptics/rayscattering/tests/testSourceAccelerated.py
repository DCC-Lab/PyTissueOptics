import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from mockito import mock, verify, when

from pytissueoptics import Vector, ScatteringScene, ScatteringMaterial, EnergyLogger, Logger
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE, CONFIG
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.opencl.CLPhotons import CLPhotons
from pytissueoptics.scene.geometry import Environment
from rayscattering.opencl import IPPTable


def tempTablePath(func):
    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as tempDir:
            IPPTable.TABLE_PATH = os.path.join(tempDir, "ipp.json")
            func(*args, **kwargs)
    return wrapper


@unittest.skipIf(not OPENCL_AVAILABLE, "Requires OpenCL to be available")
class TestSourceAccelerated(unittest.TestCase):
    SOURCE_ENV = Environment(ScatteringMaterial())
    SOURCE_POSITION = Vector(0, 0, 0)

    def setUp(self):
        self.photons = mock(CLPhotons)
        when(self.photons).setContext(...).thenReturn()
        when(self.photons).propagate(...).thenReturn()

    @patch('pytissueoptics.rayscattering.source.CLPhotons')
    def testShouldLoadPhotons(self, _CLPhotonsClassMock):
        _CLPhotonsClassMock.return_value = self.photons
        source = SinglePhotonSourceAccelerated()
        self.assertIsNotNone(source.photons)

    @tempTablePath
    @patch('pytissueoptics.rayscattering.source.CLPhotons')
    def testWhenPropagateNewExperiment_shouldWarnThatIPPWillBeEstimated(self, _CLPhotonsClassMock):
        _CLPhotonsClassMock.return_value = self.photons
        scene = self._createMockScene()
        logger = self._createMockLogger()
        source = SinglePhotonSourceAccelerated()

        with self.assertWarns(UserWarning):
            source.propagate(scene, logger)

    @tempTablePath
    @patch('pytissueoptics.rayscattering.source.CLPhotons')
    def testWhenPropagate_shouldSetCorrectPhotonContext(self, _CLPhotonsClassMock):
        _CLPhotonsClassMock.return_value = self.photons
        scene = self._createMockScene()
        logger = self._createMockLogger()
        source = SinglePhotonSourceAccelerated()

        with self.assertWarns(UserWarning):
            source.propagate(scene, logger)

        verify(self.photons).setContext(scene, self.SOURCE_ENV, logger=logger)

    @tempTablePath
    @patch('pytissueoptics.rayscattering.source.CLPhotons')
    def testGivenExperimentInIPPTable_whenPropagate_shouldUseIPPFromTable(self, _CLPhotonsClassMock):
        _CLPhotonsClassMock.return_value = self.photons
        scene = self._createMockScene()
        logger = self._createMockLogger()
        source = SinglePhotonSourceAccelerated()

        N, IPP = 10, 500
        experimentHash = hash((scene, source))
        IPPTable().updateIPP(experimentHash, N, IPP)

        source.propagate(scene, logger)

        verify(self.photons).propagate(IPP=IPP, verbose=True)

    @tempTablePath
    @patch('pytissueoptics.rayscattering.source.CLPhotons')
    @patch('pytissueoptics.rayscattering.source.Logger')
    def testGivenExperimentNotInIPPTable_whenPropagate_shouldEstimateIPP(self, _LoggerClassMock, _CLPhotonsClassMock):
        _CLPhotonsClassMock.return_value = self.photons
        source = SinglePhotonSourceAccelerated()

        # First, an IPP estimate is taken from the scene
        IPPEstimate = 80
        scene = self._createMockScene(IPPEstimate=IPPEstimate)

        # Then, a test is run with base Logger to measure IPP using 1000 photons (IPP_TEST_N_PHOTONS)
        IPPMeasuredInTest = 100
        estimationLogger = mock(Logger)
        estimationLogger.nDataPoints = CONFIG.IPP_TEST_N_PHOTONS * IPPMeasuredInTest
        _LoggerClassMock.return_value = estimationLogger

        with self.assertWarns(UserWarning):
            source.propagate(scene, self._createMockLogger())

        verify(self.photons).propagate(IPP=IPPEstimate, verbose=False)
        verify(self.photons).propagate(IPP=IPPMeasuredInTest, verbose=True)

    def _createMockScene(self, IPPEstimate=10):
        scene = mock(ScatteringScene)
        when(scene).getEstimatedIPP(...).thenReturn(IPPEstimate)
        when(scene).getEnvironmentAt(self.SOURCE_POSITION).thenReturn(self.SOURCE_ENV)
        when(scene).resetOutsideMaterial(...).thenReturn()
        when(scene).getBoundingBox().thenReturn()
        when(scene).getPolygons().thenReturn([])
        when(scene).getSolids().thenReturn([])
        return scene

    def _createMockLogger(self, nDataPoints=10):
        logger = mock(EnergyLogger)
        logger.info = {}
        logger.nDataPoints = nDataPoints
        return logger


class SinglePhotonSourceAccelerated(Source):
    def __init__(self, position=Vector(0, 0, 0)):
        super().__init__(position, N=1, useHardwareAcceleration=True)

    def getInitialPositionsAndDirections(self):
        return np.array([[0, 0, 0]]), np.array([[0, 0, 1]])

    @property
    def _hashComponents(self) -> tuple:
        return self._position,
