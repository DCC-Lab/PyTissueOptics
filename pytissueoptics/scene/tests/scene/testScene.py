import unittest

from mockito import mock, verify, when

from pytissueoptics.scene import Material
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector, BoundingBox
from pytissueoptics.scene.solids import Solid


class TestScene(unittest.TestCase):
    def setUp(self):
        self.scene = Scene()

    def testWhenAddingASolidAtAPosition_shouldPlaceTheSolidAtTheDesiredPosition(self):
        SOLID_POSITION = Vector(4, 0, 1)
        SOLID = mock(Solid)
        when(SOLID).translateTo(...).thenReturn()

        self.scene.add(SOLID, position=SOLID_POSITION)

        verify(SOLID).translateTo(SOLID_POSITION)

    def testWhenAddingASolidAtNoSpecificPosition_shouldKeepTheSolidAtItsPredefinedPosition(self):
        SOLID = mock(Solid)
        when(SOLID).translateTo(...).thenReturn()

        self.scene.add(SOLID)

        verify(SOLID, times=0).translateTo(...)

    def testWhenAddingASolidThatPartlyOverlapsWithAnotherOne_shouldNotAdd(self):
        OTHER_SOLID = self.makeSolidWith(BoundingBox([2, 5], [2, 5], [2, 5]))
        SOLID = self.makeSolidWith(BoundingBox([0, 3], [0, 3], [0, 3]))
        self.scene.add(OTHER_SOLID)

        with self.assertRaises(Exception):
            self.scene.add(SOLID)

    def testWhenAddingASolidInsideAnotherOne_shouldUpdateOutsideMaterialOfThisSolid(self):
        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True)
        SOLID = self.makeSolidWith(BoundingBox([1, 3], [1, 3], [1, 3]))
        OUTSIDE_SOLID_MATERIAL = Material()
        when(OUTSIDE_SOLID).getMaterial().thenReturn(OUTSIDE_SOLID_MATERIAL)
        self.scene.add(OUTSIDE_SOLID)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideMaterial(OUTSIDE_SOLID_MATERIAL)

    def testWhenAddingASolidOverAnotherOne_shouldUpdateOutsideMaterialOfTheOtherSolid(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]))
        self.scene.add(INSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True)
        SOLID_MATERIAL = Material()
        when(SOLID).getMaterial().thenReturn(SOLID_MATERIAL)

        self.scene.add(SOLID)

        verify(INSIDE_SOLID).setOutsideMaterial(SOLID_MATERIAL)

    def testWhenAddingASolidInsideMultipleOtherSolids_shouldUpdateOutsideMaterialOfThisSolid(self):
        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]))
        OUTSIDE_SOLID_MATERIAL = Material()
        when(OUTSIDE_SOLID).getMaterial().thenReturn(OUTSIDE_SOLID_MATERIAL)
        self.scene.add(OUTSIDE_SOLID)

        TOPMOST_OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True)
        self.scene.add(TOPMOST_OUTSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]))
        when(OUTSIDE_SOLID).contains(...).thenReturn(True)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideMaterial(OUTSIDE_SOLID_MATERIAL)

    def testWhenAddingASolidOverMultipleOtherSolids_shouldUpdateOutsideMaterialOfTheTopMostSolid(self):
        TOPMOST_INSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True)
        self.scene.add(TOPMOST_INSIDE_SOLID)

        INSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]))
        self.scene.add(INSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True)
        SOLID_MATERIAL = Material()
        when(SOLID).getMaterial().thenReturn(SOLID_MATERIAL)

        self.scene.add(SOLID)

        verify(TOPMOST_INSIDE_SOLID).setOutsideMaterial(SOLID_MATERIAL)

    def testWhenAddingASolidThatFitsInsideOneButAlsoContainsOne_shouldUpdateOutsideMaterialOfThisSolidAndTheOneInside(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]))
        self.scene.add(INSIDE_SOLID)

        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True)
        OUTSIDE_SOLID_MATERIAL = Material()
        when(OUTSIDE_SOLID).getMaterial().thenReturn(OUTSIDE_SOLID_MATERIAL)
        self.scene.add(OUTSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]))
        SOLID_MATERIAL = Material()
        when(SOLID).getMaterial().thenReturn(SOLID_MATERIAL)
        when(SOLID).contains(...).thenReturn(False).thenReturn(True)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideMaterial(OUTSIDE_SOLID_MATERIAL)
        verify(INSIDE_SOLID).setOutsideMaterial(SOLID_MATERIAL)

    def testWhenAddingASolidInsideACuboidStack_shouldRaiseNotImplementedError(self):
        pass

    def testGivenASceneWithOnlyReflectiveRaytracing_whenAddingASolidThatPartlyMergesWithAnotherOne_shouldAddTheSolid(self):
        # better support for constructive geometry?
        pass

    @staticmethod
    def makeSolidWith(bbox: BoundingBox, contains=False):
        solid = mock(Solid)
        when(solid).getBoundingBox().thenReturn(bbox)
        when(solid).getVertices().thenReturn([])
        when(solid).setOutsideMaterial(...).thenReturn()
        when(solid).getMaterial().thenReturn(Material())
        when(solid).getVertices().thenReturn([])
        when(solid).contains(...).thenReturn(contains)
        return solid
