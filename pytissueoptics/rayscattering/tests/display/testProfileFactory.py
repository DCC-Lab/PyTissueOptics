import unittest

import numpy as np
from mockito import mock, when

from pytissueoptics import View2DProjectionX
from pytissueoptics.rayscattering.display.profiles import Profile1D, ProfileFactory
from pytissueoptics.rayscattering.display.utils import Direction
from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.scene.logger import InteractionKey
from pytissueoptics.scene.solids import Cube, Sphere


class TestProfileFactory(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_CUBE = Cube(4, label="cube", material=ScatteringMaterial())
        self.TEST_SPHERE = Sphere(1, order=1, label="sphere", material=ScatteringMaterial())
        self.TEST_SCENE = ScatteringScene([self.TEST_CUBE, self.TEST_SPHERE])
        self.DEFAULT_BIN_SIZE = 0.1
        self.DEFAULT_BIN_SIZE_3D = (0.1, 0.1, 0.1)
        self.INFINITE_LIMITS = [(-10, 10), (-10, 10), (-10, 10)]
        self.TEST_SCENE_LIMITS = self.TEST_SCENE.getBoundingBox().xyzLimits
        self.TEST_SCENE_LIMITS = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]

    def setUp(self):
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, defaultBinSize=self.DEFAULT_BIN_SIZE)
        self._fillLoggerData()
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

    def _fillLoggerData(self):
        testDataDiag = np.zeros((4, 4))
        for i in range(4):
            testDataDiag[i, 0] = i
            testDataDiag[i, 1:] = -1.5 + i
        self.TEST_LOGGER.logDataPointArray(testDataDiag, InteractionKey("cube"))
        self.TEST_LOGGER.logDataPointArray(testDataDiag, InteractionKey("sphere"))

    def testWhenCreateDefaultProfile_shouldSetProfileLimitsToSceneLimits(self):
        profile = self.profileFactory.create(Direction.Z_POS)

        sceneLimitsAlongProfile = self.TEST_SCENE_LIMITS[profile.horizontalDirection.axis]
        self.assertEqual(sceneLimitsAlongProfile, profile.limits)

    def testWhenCreateDefaultProfile_shouldSetProfileBinsWithDefaultLoggerBinSize(self):
        profile = self.profileFactory.create(Direction.Z_POS)

        nBins = int((profile.limits[1] - profile.limits[0]) / self.DEFAULT_BIN_SIZE)
        self.assertEqual(nBins, profile.data.size)

    def testWhenCreateProfileWithSolidLabel_shouldSetProfileLimitsToSolidLimits(self):
        profile = self.profileFactory.create(Direction.Z_POS, solidLabel=self.TEST_CUBE.getLabel())
        solidLimitsAlongProfile = tuple(self.TEST_CUBE.getBoundingBox().xyzLimits[profile.horizontalDirection.axis])
        self.assertEqual(solidLimitsAlongProfile, profile.limits)

    def testWhenCreateProfileWithLimits_shouldSetProfileLimitsToGivenLimits(self):
        profile = self.profileFactory.create(Direction.Z_POS, limits=(-1, 1))
        self.assertEqual((-1, 1), profile.limits)

    def testWhenCreateProfileWithBinSize_shouldSetProfileBinsWithGivenBinSize(self):
        profile = self.profileFactory.create(Direction.Z_POS, binSize=0.5)
        nBins = int((profile.limits[1] - profile.limits[0]) / 0.5)
        self.assertEqual(nBins, profile.data.size)

    def testGivenInfiniteScene_whenCreateDefaultProfile_shouldSetProfileLimitsToDefaultLoggerInfiniteLimits(self):
        infiniteScene = mock(ScatteringScene)
        when(infiniteScene).getBoundingBox().thenReturn(None)
        self.TEST_LOGGER = EnergyLogger(infiniteScene, infiniteLimits=self.INFINITE_LIMITS, views=[])
        self.profileFactory = ProfileFactory(infiniteScene, self.TEST_LOGGER)

        profile = self.profileFactory.create(Direction.Z_POS)

        expectedLimits = self.INFINITE_LIMITS[profile.horizontalDirection.axis]
        self.assertEqual(expectedLimits, profile.limits)

    def testWhenCreateProfile_shouldExtractProfileDataFromLogger(self):
        expectedDataPerSolid = np.array([0, 1, 2, 3])
        expectedData = expectedDataPerSolid * 2

        for direction in [Direction.X_POS, Direction.Y_POS, Direction.Z_POS]:
            profile = self.profileFactory.create(direction, binSize=1)
            self.assertTrue(np.array_equal(expectedData, profile.data))

    def testWhenCreateProfileWithNegativeDirection_shouldExtractSameProfileDataFromLogger(self):
        expectedDataPerSolid = np.array([0, 1, 2, 3])
        expectedData = expectedDataPerSolid * 2

        for direction in [Direction.X_NEG, Direction.Y_NEG, Direction.Z_NEG]:
            profile = self.profileFactory.create(direction, binSize=1)
            self.assertTrue(np.array_equal(expectedData, profile.data))

    def testWhenCreateProfileOfSurfaceEnergyLeaving_shouldOnlyExtractSurfaceEnergyLeavingDataFromLogger(self):
        # Logging a point leaving the surface in the middle of the last profile bin (1 to 2)
        #  and a point entering the surface in the middle of the first profile bin (-2 to -1)
        surfaceData = np.array([[1, 1.5, 1.5, 1.5], [-1, -1.5, -1.5, -1.5]])
        self.TEST_LOGGER.logDataPointArray(surfaceData, InteractionKey("cube", "top"))

        profile = self.profileFactory.create(
            Direction.Z_POS, solidLabel="cube", surfaceLabel="top", surfaceEnergyLeaving=True, binSize=1
        )

        expectedData = np.array([0, 0, 0, 1])
        self.assertTrue(np.array_equal(expectedData, profile.data))

    def testWhenCreateProfileOfSurfaceEnergyEntering_shouldOnlyExtractSurfaceEnergyEnteringDataFromLogger(self):
        surfaceData = np.array([[1, 1.5, 1.5, 1.5], [-1, -1.5, -1.5, -1.5]])
        self.TEST_LOGGER.logDataPointArray(surfaceData, InteractionKey("cube", "top"))

        profile = self.profileFactory.create(
            Direction.Z_POS, solidLabel="cube", surfaceLabel="top", surfaceEnergyLeaving=False, binSize=1
        )

        expectedData = np.array([1, 0, 0, 0])
        self.assertTrue(np.array_equal(expectedData, profile.data))

    def testWhenCreateProfileWithBadLabelCapitalization_shouldCorrectLabels(self):
        self.TEST_LOGGER.logDataPointArray(np.array([[1, 0, 0, 0]]), InteractionKey("cube", "top"))
        profile = self.profileFactory.create(
            Direction.Z_POS, solidLabel="Cube", surfaceLabel="Top", surfaceEnergyLeaving=True
        )
        self.assertEqual(1, profile.data.sum())

    def testWhenCreateProfile_shouldSetProperName(self):
        self.TEST_LOGGER.logDataPointArray(np.array([[1, 0, 0, 0]]), InteractionKey("cube", "top"))
        profile = self.profileFactory.create(
            Direction.Z_POS, solidLabel="cube", surfaceLabel="top", surfaceEnergyLeaving=True
        )
        self.assertEqual("Energy profile along z of cube surface top (leaving)", profile.name)

    def testGivenEmptyLogger_whenCreateProfile_shouldReturnEmptyProfile(self):
        emptyLogger = EnergyLogger(self.TEST_SCENE)
        self.profileFactory = ProfileFactory(self.TEST_SCENE, emptyLogger)

        profile = self.profileFactory.create(Direction.Z_POS)

        self.assertIsInstance(profile, Profile1D)
        self.assertEqual(0, profile.data.sum())

    def testGiven2DLogger_whenCreateDefaultProfile_shouldExtractProfileFromLoggerData(self):
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, keep3D=False, defaultBinSize=1.0)
        self._fillLoggerData()
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

        expectedDataPerSolid = np.array([0, 1, 2, 3])
        expectedData = expectedDataPerSolid * 2

        for direction in [Direction.X_POS, Direction.Y_POS, Direction.Z_POS]:
            profile = self.profileFactory.create(direction, binSize=1)
            self.assertTrue(np.array_equal(expectedData, profile.data))

    def testGiven2DLogger_whenCreateSolidProfile_shouldExtractSolidProfileFromLoggerData(self):
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, keep3D=False, defaultBinSize=1.0)
        self._fillLoggerData()
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

        expectedDataPerSolid = np.array([0, 1, 2, 3])

        for direction in [Direction.X_POS, Direction.Y_POS, Direction.Z_POS]:
            profile = self.profileFactory.create(direction, binSize=1, solidLabel="cube")
            self.assertTrue(np.array_equal(expectedDataPerSolid, profile.data))

    def testGiven2DLogger_whenCreateSurfaceProfile_shouldExtractSolidProfileFromLoggerData(self):
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, keep3D=False, defaultBinSize=1.0)
        surfaceData = np.array([[1, 1.5, 1.5, 1.5], [-1, -1.5, -1.5, -1.5]])
        self.TEST_LOGGER.logDataPointArray(surfaceData, InteractionKey("cube", "cube_top"))
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

        expectedSurfaceDataLeaving = np.array([0, 0, 0, 1])
        expectedSurfaceDataEntering = np.array([1, 0, 0, 0])

        profileDirection = Direction.Z_POS  # All directions are allowed except for surface axis (Y)
        profile = self.profileFactory.create(
            profileDirection, binSize=1, solidLabel="cube", surfaceLabel="top", surfaceEnergyLeaving=True
        )
        self.assertTrue(np.array_equal(expectedSurfaceDataLeaving, profile.data))
        profile = self.profileFactory.create(
            profileDirection, binSize=1, solidLabel="cube", surfaceLabel="top", surfaceEnergyLeaving=False
        )
        self.assertTrue(np.array_equal(expectedSurfaceDataEntering, profile.data))

    def testGiven2DLoggerWithASingleView_whenCreateProfileAlongThisViewWithSameProperties_shouldExtractProfile(self):
        singleView = View2DProjectionX()
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, keep3D=False, defaultBinSize=1.0, views=[singleView])
        self._fillLoggerData()
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

        profile = self.profileFactory.create(Direction.Z_POS, binSize=1)
        expectedData = np.array([0, 1, 2, 3]) * 2
        self.assertTrue(np.array_equal(expectedData, profile.data))

    def testGiven2DLoggerWithASingleView_whenCreateProfileAlongThisViewWithDifferentBinSize_shouldRaiseException(self):
        singleView = View2DProjectionX()
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, keep3D=False, defaultBinSize=1.0, views=[singleView])
        self._fillLoggerData()
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

        with self.assertRaises(RuntimeError):
            self.profileFactory.create(Direction.Z_POS, binSize=0.5)

    def testGiven2DLoggerWithASingleView_whenCreateProfileAlongThisViewWithDifferentLimits_shouldRaiseException(self):
        singleView = View2DProjectionX()
        self.TEST_LOGGER = EnergyLogger(self.TEST_SCENE, keep3D=False, defaultBinSize=1.0, views=[singleView])
        self._fillLoggerData()
        self.profileFactory = ProfileFactory(self.TEST_SCENE, self.TEST_LOGGER)

        with self.assertRaises(RuntimeError):
            self.profileFactory.create(Direction.Z_POS, binSize=1.0, limits=(0, 2))
