import unittest

from mockito import mock, verify, when

from pytissueoptics.scene import Material
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector, BoundingBox
from pytissueoptics.scene.solids import Solid


class TestScene(unittest.TestCase):
    THE_POSITION = Vector(4, 0, 1)

    def setUp(self):
        self.scene = Scene()

    def testWhenAddingASolidAtAPosition_shouldPlaceTheSolidAtTheDesiredPosition(self):
        THE_SOLID = mock(Solid)
        when(THE_SOLID).translateTo(...).thenReturn()

        self.scene.add(THE_SOLID, position=self.THE_POSITION)

        verify(THE_SOLID).translateTo(self.THE_POSITION)

    def testWhenAddingASolidAtNoSpecificPosition_shouldKeepTheSolidAtItsPredefinedPosition(self):
        THE_SOLID = mock(Solid)
        when(THE_SOLID).translateTo(...).thenReturn()

        self.scene.add(THE_SOLID)

        verify(THE_SOLID, times=0).translateTo(...)

    def testWhenAddingASolidThatPartlyOverlapsWithAnotherOne_shouldNotAdd(self):
        OTHER_SOLID = self.makeSolidWithBBox(BoundingBox([2, 5], [2, 5], [2, 5]))
        THE_SOLID = self.makeSolidWithBBox(BoundingBox([0, 3], [0, 3], [0, 3]))
        self.scene.add(OTHER_SOLID)

        with self.assertRaises(Exception):
            self.scene.add(THE_SOLID)

    def testWhenAddingASolidInsideAnotherOne_shouldUpdateOutsideMaterialOfThisSolid(self):
        OTHER_SOLID = self.makeSolidWithBBox(BoundingBox([0, 5], [0, 5], [0, 5]))
        THE_SOLID = self.makeSolidWithBBox(BoundingBox([1, 3], [1, 3], [1, 3]))
        otherMaterial = Material()
        when(OTHER_SOLID).getMaterial().thenReturn(otherMaterial)
        when(OTHER_SOLID).contains(...).thenReturn(True)
        when(THE_SOLID).contains(...).thenReturn(False)
        self.scene.add(OTHER_SOLID)

        self.scene.add(THE_SOLID)

        verify(THE_SOLID).setOutsideMaterial(otherMaterial)

    def testWhenAddingASolidOverAnotherOne_shouldUpdateOutsideMaterialOfTheOtherSolid(self):
        OTHER_SOLID = self.makeSolidWithBBox(BoundingBox([0, 5], [0, 5], [0, 5]))
        THE_SOLID = self.makeSolidWithBBox(BoundingBox([-1, 6], [-1, 6], [-1, 6]))
        solidMaterial = Material()
        when(THE_SOLID).getMaterial().thenReturn(solidMaterial)
        when(OTHER_SOLID).contains(...).thenReturn(False)
        when(THE_SOLID).contains(...).thenReturn(True)
        self.scene.add(OTHER_SOLID)

        self.scene.add(THE_SOLID)

        verify(OTHER_SOLID).setOutsideMaterial(solidMaterial)

    def testWhenAddingASolidInsideMultipleOtherSolids_shouldUpdateOutsideMaterialOfThisSolid(self):
        pass

    def testWhenAddingASolidOverMultipleOtherSolids_shouldUpdateOutsideMaterialOfTheTopMostSolid(self):
        pass

    def testWhenAddingASolidThatFitsInsideOneButAlsoContainsOne_shouldUpdateOutsideMaterialOfThisSolidAndTheOneInside(self):
        pass

    def testWhenAddingASolidInsideACuboidStack_shouldRaiseNotImplementedError(self):
        pass

    def testGivenASceneWithOnlyReflectiveRaytracing_whenAddingASolidThatPartlyMergesWithAnotherOne_shouldAddTheSolid(self):
        # better support for constructive geometry?
        pass

    @staticmethod
    def makeSolidWithBBox(bbox: BoundingBox):
        solid = mock(Solid)
        when(solid).getBoundingBox().thenReturn(bbox)
        when(solid).getVertices().thenReturn([])
        when(solid).setOutsideMaterial(...).thenReturn()
        when(solid).getVertices().thenReturn([])
        return solid
