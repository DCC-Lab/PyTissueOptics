import unittest

from mockito import mock, when, verify

from pytissueoptics.scene.solids import Sphere, Cube
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.display.views import *


class TestViewFactory(unittest.TestCase):
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
        self.viewFactory = ViewFactory(self.TEST_SCENE, self.DEFAULT_BIN_SIZE, self.INFINITE_LIMITS)

    def testWhenBuildNoViews_shouldReturnNothing(self):
        views = self.viewFactory.build(None)
        self.assertEqual(0, len(views))

    def testWhenBuildAViewWithoutSolidLabel_shouldSetContextWithSceneLimitsAndDefaultBinSize(self):
        sceneView = mock(View2D)
        sceneView.solidLabel = None
        when(sceneView).setContext(...).thenReturn()

        self.viewFactory.build([sceneView])

        verify(sceneView).setContext(limits3D=self.TEST_SCENE_LIMITS, binSize3D=self.DEFAULT_BIN_SIZE_3D)

    def testWhenBuildAViewWithSolidLabel_shouldSetContextWithSolidLimits(self):
        sceneView = mock(View2D)
        sceneView.solidLabel = self.TEST_CUBE.getLabel()
        sceneView.surfaceLabel = None
        when(sceneView).setContext(...).thenReturn()

        self.viewFactory.build([sceneView])

        solidLimits = self.TEST_CUBE.getBoundingBox().xyzLimits
        solidLimits = [(d[0], d[1]) for d in solidLimits]
        verify(sceneView).setContext(limits3D=solidLimits, binSize3D=self.DEFAULT_BIN_SIZE_3D)

    def testWhenBuildAViewWithInfiniteScene_shouldSetContextWithDefaultInfiniteLimits(self):
        sceneView = mock(View2D)
        sceneView.solidLabel = None
        when(sceneView).setContext(...).thenReturn()
        infiniteScene = mock(ScatteringScene)
        when(infiniteScene).getBoundingBox().thenReturn(None)
        self.viewFactory = ViewFactory(infiniteScene, self.DEFAULT_BIN_SIZE_3D, self.INFINITE_LIMITS)

        self.viewFactory.build([sceneView])

        verify(sceneView).setContext(limits3D=self.INFINITE_LIMITS, binSize3D=self.DEFAULT_BIN_SIZE_3D)

    def testWhenBuildASceneViewGroup_shouldCreateAndReturnTheDefaultProjectionViews(self):
        sceneViews = self.viewFactory.build(ViewGroup.SCENE)

        self.assertEqual(3, len(sceneViews))
        viewTypes = [View2DProjectionX, View2DProjectionY, View2DProjectionZ]
        for i, view in enumerate(sceneViews):
            self.assertIsNone(view.solidLabel)
            self.assertIsInstance(view, viewTypes[i])

    def testWhenBuildASolidsViewGroup_shouldCreateAndReturnTheDefaultProjectionViewsForEachSolid(self):
        solidViews = self.viewFactory.build(ViewGroup.SOLIDS)

        self.assertEqual(6, len(solidViews))
        viewTypes = [View2DProjectionX, View2DProjectionY, View2DProjectionZ]
        solidLabels = [self.TEST_CUBE.getLabel(), self.TEST_SPHERE.getLabel()]
        for i, view in enumerate(solidViews):
            self.assertEqual(solidLabels[i // 3], view.solidLabel)
            self.assertIsInstance(view, viewTypes[i % 3])

    def testWhenBuildASurfacesLeavingViewGroup_shouldCreateAndReturnTheDefaultLeavingSurfaceViewForEachSolidSurface(self):
        # These include a single view for the surface of the sphere taken along Z,
        #  and another view of the energy leaving from the cube and into the sphere
        surfaceLeavingViews = self.viewFactory.build(ViewGroup.SURFACES_LEAVING)

        expectedViews = [View2DSurfaceX("cube", "cube_left"),
                         View2DSurfaceX("cube", "cube_right"),
                         View2DSurfaceY("cube", "cube_bottom"),
                         View2DSurfaceY("cube", "cube_top"),
                         View2DSurfaceZ("cube", "cube_front"),
                         View2DSurfaceZ("cube", "cube_back"),
                         View2DSurfaceZ("cube", "sphere_ellipsoid"),
                         View2DSurfaceZ("sphere", "sphere_ellipsoid")]
        expectedViews[1].flip()
        expectedViews[2].flip()
        expectedViews[5].flip()

        self.viewFactory.build(expectedViews)

        self.assertEqual(len(expectedViews), len(surfaceLeavingViews))

        for i in range(len(expectedViews)):
            self.assertTrue(expectedViews[i].isEqualTo(surfaceLeavingViews[i]))

    def testWhenBuildASurfacesEnteringViewGroup_shouldCreateAndReturnTheDefaultEnteringSurfaceViewForEachSolidSurface(self):
        surfaceEnteringViews = self.viewFactory.build(ViewGroup.SURFACES_ENTERING)

        expectedViews = [View2DSurfaceX("cube", "cube_left", surfaceEnergyLeaving=False),
                         View2DSurfaceX("cube", "cube_right", surfaceEnergyLeaving=False),
                         View2DSurfaceY("cube", "cube_bottom", surfaceEnergyLeaving=False),
                         View2DSurfaceY("cube", "cube_top", surfaceEnergyLeaving=False),
                         View2DSurfaceZ("cube", "cube_front", surfaceEnergyLeaving=False),
                         View2DSurfaceZ("cube", "cube_back", surfaceEnergyLeaving=False),
                         View2DSurfaceZ("cube", "sphere_ellipsoid", surfaceEnergyLeaving=False),
                         View2DSurfaceZ("sphere", "sphere_ellipsoid", surfaceEnergyLeaving=False)]
        expectedViews[1].flip()
        expectedViews[2].flip()
        expectedViews[5].flip()

        self.viewFactory.build(expectedViews)

        self.assertEqual(len(expectedViews), len(surfaceEnteringViews))

        for i in range(len(expectedViews)):
            self.assertTrue(expectedViews[i].isEqualTo(surfaceEnteringViews[i]))

    def testWhenBuildASurfacesViewGroup_shouldCreateBothDefaultEnteringAndLeavingSurfaceViews(self):
        surfaceViews = self.viewFactory.build(ViewGroup.SURFACES)

        self.assertEqual(16, len(surfaceViews))
        self.assertEqual(8, len([v for v in surfaceViews if v.surfaceEnergyLeaving]))
        self.assertEqual(8, len([v for v in surfaceViews if not v.surfaceEnergyLeaving]))
