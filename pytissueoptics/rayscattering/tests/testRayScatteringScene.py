import unittest

from pytissueoptics import RayScatteringScene, ScatteringMaterial, Cuboid


class TestRayScatteringScene(unittest.TestCase):
    def testWhenAddingASolidWithAScatteringMaterial_shouldAddSolidToTheScene(self):
        scene = RayScatteringScene([Cuboid(1, 1, 1, material=ScatteringMaterial())])
        self.assertEqual(len(scene.solids), 1)

    def testWhenAddingASolidWithNoScatteringMaterialDefined_shouldRaiseException(self):
        with self.assertRaises(Exception):
            RayScatteringScene([Cuboid(1, 1, 1)])
        with self.assertRaises(Exception):
            RayScatteringScene([Cuboid(1, 1, 1, material="Not a scattering material")])
