import unittest


class TestScene(unittest.TestCase):
    def testWhenAddingASolid_shouldAddItToTheSceneAtTheDesiredPosition(self):
        pass

    def testWhenAddingASolidWithNonZeroPositionToADesiredPosition_shouldWarnAboutRepositioning(self):
        pass

    def testWhenAddingASolidInsideAnotherOne_shouldUpdateOutsideMaterialOfThisSolid(self):
        # when adding solid, check for intersections (if 2 bbox touches,
        # assert one is contained in the other and update outside material)
        pass

    def testWhenAddingASolidOverAnotherOne_shouldUpdateOutsideMaterialOfTheOtherSolid(self):
        pass

    def testWhenAddingASolidThatPartlyMergesWithAnotherOne_shouldNotAdd(self):
        pass

    def testWhenAddingASolidInsideACuboidStack_shouldRaiseNotImplementedError(self):
        pass

    def testGivenASceneWithOnlyReflectiveRaytracing_whenAddingASolidThatPartlyMergesWithAnotherOne_shouldAddTheSolidToTheSceneWithoutAffectingExistingSolid(self):
        # better support for constructive geometry?
        pass
