import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cube, SolidFactory


class TestSolidGroup(unittest.TestCase):
    def setUp(self):
        self.material = "A Material"
        self.position = Vector(0, 0, 10)
        self.cuboidSurfaceLabels = ["left", "right", "bottom", "top", "front", "back"]
        self.cuboid1 = Cube(edge=2, position=Vector(2, 0, 0), material=self.material, label="cuboid1")
        self.cuboid2 = Cube(edge=2, position=Vector(-2, 0, 0), material=self.material, label="cuboid2")
        self.solidGroup = SolidFactory().fromSolids([self.cuboid1, self.cuboid2], position=self.position)

    def testShouldHaveSurfacesOfAllInputSolids(self):
        groupSurfaceLabels = self.solidGroup.surfaceLabels

        self.assertEqual(12, len(groupSurfaceLabels))
        for cuboid in [self.cuboid1, self.cuboid2]:
            for surfaceLabel in self.cuboidSurfaceLabels:
                self.assertIn(f"{cuboid.getLabel()}_{surfaceLabel}", groupSurfaceLabels)

    def testShouldMoveCentroidToBeAtDesiredPosition(self):
        self.assertEqual(self.solidGroup.position, self.position)

    def testGivenSolidsWithTheSameLabel_shouldIncrementSolidLabels(self):
        CUBOID_LABEL = "cuboid"
        cuboid1 = Cube(edge=1, position=Vector(2, 0, 0), material=self.material, label=CUBOID_LABEL)
        cuboid2 = Cube(edge=1, position=Vector(-2, 0, 0), material=self.material, label=CUBOID_LABEL)
        cuboid3 = Cube(edge=1, position=Vector(0, 0, 0), material=self.material, label=CUBOID_LABEL)
        solidGroup = SolidFactory().fromSolids([cuboid1, cuboid2, cuboid3], position=self.position)

        groupSurfaceLabels = solidGroup.surfaceLabels
        self.assertEqual(18, len(groupSurfaceLabels))
        for expectedCuboidLabel in [CUBOID_LABEL, f"{CUBOID_LABEL}_2", f"{CUBOID_LABEL}_3"]:
            for surfaceLabel in self.cuboidSurfaceLabels:
                self.assertIn(f"{expectedCuboidLabel}_{surfaceLabel}", groupSurfaceLabels)
