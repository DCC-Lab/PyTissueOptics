import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import SolidGroupMerge, Cube


class TestSolidGroupMerge(unittest.TestCase):
    def setUp(self):
        self.material = "A Material"
        self.position = Vector(0, 0, 0)
        cuboid1 = Cube(edge=2, position=Vector(2, 0, 0), material=self.material, label="cuboid")
        cuboid2 = Cube(edge=2, position=Vector(-2, 0, 0), material=self.material, label="cuboid")
        self.solidGroup = SolidGroupMerge([cuboid1, cuboid2])

    def testShouldHave12SurfacesWithAppropriateLabels(self):
        self.assertEqual(len(self.solidGroup.surfaceLabels), 12)
        self.assertEqual(self.solidGroup.surfaceLabels[0], "cuboid_left")
        self.assertEqual(self.solidGroup.surfaceLabels[1], "cuboid_right")
        self.assertEqual(self.solidGroup.surfaceLabels[2], "cuboid_bottom")
        self.assertEqual(self.solidGroup.surfaceLabels[3], "cuboid_top")
        self.assertEqual(self.solidGroup.surfaceLabels[4], "cuboid_front")
        self.assertEqual(self.solidGroup.surfaceLabels[5], "cuboid_back")

        self.assertEqual(self.solidGroup.surfaceLabels[6], "cuboid_0_left")
        self.assertEqual(self.solidGroup.surfaceLabels[7], "cuboid_0_right")
        self.assertEqual(self.solidGroup.surfaceLabels[8], "cuboid_0_bottom")
        self.assertEqual(self.solidGroup.surfaceLabels[9], "cuboid_0_top")
        self.assertEqual(self.solidGroup.surfaceLabels[10], "cuboid_0_front")
        self.assertEqual(self.solidGroup.surfaceLabels[11], "cuboid_0_back")

    def testShouldMoveCentroidToBeAtDesiredPosition(self):
        self.assertEqual(self.solidGroup.getSolidsCentroid(), self.position)
