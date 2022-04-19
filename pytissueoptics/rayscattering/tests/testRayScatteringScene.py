import unittest

from mockito import mock, when, verify

from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Material
from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.solids import Solid


class TestRayScatteringScene(unittest.TestCase):
    def testWhenSetWorldMaterial_shouldSetOutsideMaterialOfAllItsSolids(self):
        solid = mock(Solid)
        when(solid).setOutsideEnvironment(...).thenReturn()
        when(solid).getLabel().thenReturn()
        tissue = RayScatteringScene([solid])
        worldMaterial = Material()

        tissue.setWorldMaterial(worldMaterial)

        verify(solid).setOutsideEnvironment(Environment(worldMaterial))
