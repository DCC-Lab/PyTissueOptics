import unittest

from mockito import mock, when, verify

from pytissueoptics.rayscattering.tissues import Tissue
from pytissueoptics.scene import Material
from pytissueoptics.scene.solids import Solid


class TestTissue(unittest.TestCase):
    def testWhenSetWorldMaterial_shouldSetOutsideMaterialOfAllItsSolids(self):
        solid = mock(Solid)
        when(solid).setOutsideMaterial(...).thenReturn()
        tissue = Tissue([solid])
        worldMaterial = Material()

        tissue.setWorldMaterial(worldMaterial)

        verify(solid).setOutsideMaterial(worldMaterial)
