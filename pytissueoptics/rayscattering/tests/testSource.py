import unittest

from mockito import mock, when, verify

from pytissueoptics.rayscattering import PencilSource, Photon
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Vector
from pytissueoptics.scene.geometry import Environment


class TestSource(unittest.TestCase):
    def setUp(self):
        self.photon = self._createPhoton()
        self.source = Source(position=Vector(), direction=Vector(), photons=[self.photon])

    def testWhenPropagate_shouldSetWorldMaterialInTissue(self):
        worldMaterial = ScatteringMaterial()
        tissue = self._createTissue()

        self.source.propagate(tissue, worldMaterial=worldMaterial)

        verify(tissue).setOutsideMaterial(worldMaterial)

    def testWhenPropagate_shouldSetInitialPhotonMaterialAsWorldMaterial(self):
        worldMaterial = ScatteringMaterial()
        self.source.propagate(self._createTissue(), worldMaterial=worldMaterial)
        verify(self.photon).setContext(Environment(worldMaterial), ...)

    def testWhenPropagate_shouldPropagateAllPhotons(self):
        self.source.propagate(self._createTissue(), worldMaterial=ScatteringMaterial())
        verify(self.photon).propagate()

    @staticmethod
    def _createTissue():
        tissue = mock(RayScatteringScene)
        when(tissue).setOutsideMaterial(...).thenReturn()
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
        pencilSource = PencilSource(direction=sourceDirection, N=10)
        for photon in pencilSource.photons:
            self.assertEqual(sourceDirection, photon.direction)

    def testShouldHavePhotonsAllPositionedAtTheSourcePosition(self):
        sourcePosition = Vector(3, 3, 0)
        pencilSource = PencilSource(position=sourcePosition, N=10)
        for photon in pencilSource.photons:
            self.assertEqual(sourcePosition, photon.position)
