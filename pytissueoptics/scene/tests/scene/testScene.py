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
        otherBBox = BoundingBox([2, 5], [2, 5], [2, 5])
        OTHER_SOLID = mock(Solid, {"bbox": otherBBox})
        solidBBox = BoundingBox([0, 3], [0, 3], [0, 3])
        THE_SOLID = mock(Solid, {"bbox": solidBBox})
        self.scene.add(OTHER_SOLID)

        with self.assertRaises(Exception):
            self.scene.add(THE_SOLID)

    def testWhenAddingASolidInsideAnotherOne_shouldUpdateOutsideMaterialOfThisSolid(self):
        OTHER_SOLID = mock(Solid)
        otherBBox = BoundingBox([0, 5], [0, 5], [0, 5])
        otherMaterial = Material()
        when(OTHER_SOLID).getBoundingBox().thenReturn(otherBBox)
        when(OTHER_SOLID).getMaterial().thenReturn(otherMaterial)
        self.scene.add(OTHER_SOLID)

        THE_SOLID = mock(Solid)
        solidBBox = BoundingBox([1, 3], [1, 3], [1, 3])
        when(THE_SOLID).getBoundingBox().thenReturn(solidBBox)
        when(THE_SOLID).setOutsideMaterial(...).thenReturn()

        when(OTHER_SOLID).contains(...).thenReturn(True)
        when(THE_SOLID).contains(...).thenReturn(False)

        self.scene.add(THE_SOLID)

        verify(THE_SOLID).setOutsideMaterial(otherMaterial)

    def testWhenAddingASolidOverAnotherOne_shouldUpdateOutsideMaterialOfTheOtherSolid(self):
        pass

    def testWhenAddingASolidInsideACuboidStack_shouldRaiseNotImplementedError(self):
        pass

    def testGivenASceneWithOnlyReflectiveRaytracing_whenAddingASolidThatPartlyMergesWithAnotherOne_shouldAddTheSolidToTheSceneWithoutAffectingExistingSolid(self):
        # better support for constructive geometry?
        pass
