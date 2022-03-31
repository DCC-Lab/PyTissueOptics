import unittest

from mockito import mock, when, verify

from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Material
from pytissueoptics.scene.solids import Solid


class TestRayScatteringScene(unittest.TestCase):
    def testWhenSetWorldMaterial_shouldSetOutsideMaterialOfAllItsSolids(self):
        solid = mock(Solid)
        when(solid).setOutsideMaterial(...).thenReturn()
        tissue = RayScatteringScene([solid])
        worldMaterial = Material()

        tissue.setWorldMaterial(worldMaterial)

        verify(solid).setOutsideMaterial(worldMaterial)
