import math
import unittest
from unittest.mock import patch

from mockito import mock, verify, when

from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene.solids import Cuboid
from pytissueoptics.scene.viewer import MayaviViewer


def patchMayaviShow(func):
    for module in ['show', 'gcf', 'figure', 'clf', 'triangular_mesh']:
        func = patch('mayavi.mlab.' + module)(func)
    return func


class TestScatteringScene(unittest.TestCase):
    def testWhenAddingASolidWithAScatteringMaterial_shouldAddSolidToTheScene(self):
        scene = ScatteringScene([Cuboid(1, 1, 1, material=ScatteringMaterial())])
        self.assertEqual(len(scene.solids), 1)

    def testWhenAddingASolidWithNoScatteringMaterialDefined_shouldRaiseException(self):
        with self.assertRaises(Exception):
            ScatteringScene([Cuboid(1, 1, 1)])
        with self.assertRaises(Exception):
            ScatteringScene([Cuboid(1, 1, 1, material="Not a scattering material")])

    def testWhenAddToViewer_shouldAddAllSolidsToViewer(self):
        scene = ScatteringScene([Cuboid(1, 1, 1, material=ScatteringMaterial())])
        viewer = mock(MayaviViewer)
        when(viewer).add(...).thenReturn()

        scene.addToViewer(viewer)

        verify(viewer).add(*scene.solids, ...)

    @patchMayaviShow
    def testWhenDisplay_shouldDisplayWithMayaviViewer(self, mockShow, *args):
        scene = ScatteringScene([Cuboid(1, 1, 1, material=ScatteringMaterial())])
        scene.display()

        mockShow.assert_called_once()

    def testShouldHaveIPPEstimationUsingMeanAlbedoInInfiniteMedium(self):
        material1 = ScatteringMaterial(mu_s=1, mu_a=0.7, g=0.9)
        material2 = ScatteringMaterial(mu_s=8, mu_a=1, g=0.9)
        meanAlbedo = (material1.getAlbedo() + material2.getAlbedo()) / 2
        weightThreshold = 0.0001

        scene = ScatteringScene([Cuboid(1, 1, 1, material=material2)], worldMaterial=material1)

        estimation = scene.getEstimatedIPP(weightThreshold)
        expectedEstimation = -math.log(weightThreshold) / meanAlbedo
        self.assertEqual(expectedEstimation, estimation)
