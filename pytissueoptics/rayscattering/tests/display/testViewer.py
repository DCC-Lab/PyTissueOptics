import unittest
from unittest.mock import Mock, patch

import numpy as np
from mockito import ANY, mock, verify, when

from pytissueoptics import Direction, View2DProjectionX, ViewGroup
from pytissueoptics.rayscattering.display.profiles import Profile1D, ProfileFactory
from pytissueoptics.rayscattering.display.viewer import PointCloudStyle, Viewer, Visibility
from pytissueoptics.rayscattering.display.views import View2D
from pytissueoptics.rayscattering.energyLogging import EnergyLogger, PointCloud
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.viewer import Abstract3DViewer


class TestViewer(unittest.TestCase):
    def setUp(self):
        self.scene = mock(ScatteringScene)
        self.source = mock(Source)
        when(self.source).addToViewer(...).thenReturn()
        when(self.scene).addToViewer(...).thenReturn()
        self.logger = mock(EnergyLogger)
        self.logger.defaultBinSize = 1
        self.logger.infiniteLimits = ((-10, 10), (-10, 10), (-10, 10))
        self.logger.has3D = True
        self.logger.info = {"photonCount": 0, "sourceSolidLabel": None}
        self.viewer = Viewer(self.scene, self.source, self.logger)

        self.mock3DViewer = Mock(spec=Abstract3DViewer)
        p = patch("pytissueoptics.rayscattering.display.viewer.get3DViewer", return_value=self.mock3DViewer)
        self.addCleanup(p.stop)
        p.start()

    def testGivenViewerWithBaseLogger_shouldRaiseException(self):
        with self.assertRaises(AssertionError):
            self.viewer = Viewer(self.scene, self.source, Logger())

    def testWhenListViews_shouldListTheLoggerViews(self):
        when(self.logger).listViews().thenReturn()
        self.viewer.listViews()
        verify(self.logger, times=1).listViews()

    @patch("pytissueoptics.rayscattering.statistics.statistics.Stats.report")
    def testWhenReportStats_shouldReportStats(self, mockReport):
        self.viewer.reportStats()
        mockReport.assert_called_once()

    def testWhenShow1D_shouldCreate1DProfileAndDisplay(self):
        mockProfileFactory = mock(ProfileFactory)
        mockProfile = mock(Profile1D)
        when(mockProfileFactory).create(...).thenReturn(mockProfile)
        when(mockProfile).show(...).thenReturn()
        self.viewer._profileFactory = mockProfileFactory

        self.viewer.show1D(Direction.Z_POS)

        verify(mockProfileFactory, times=1).create(Direction.Z_POS, ...)
        verify(mockProfile, times=1).show(...)

    def testWhenShow2D_shouldShow2DViewFromLogger(self):
        when(self.logger).showView(...).thenReturn()
        view = View2DProjectionX()

        self.viewer.show2D(view)

        verify(self.logger, times=1).showView(view=view, viewIndex=None, logScale=True, colormap=ANY(str))

    def testWhenShow2DWithIndex_shouldShow2DViewFromLogger(self):
        when(self.logger).showView(...).thenReturn()

        self.viewer.show2D(viewIndex=1)

        verify(self.logger, times=1).showView(view=None, viewIndex=1, logScale=True, colormap=ANY(str))

    def testWhenShow2DAllViews_shouldShowAllViews(self):
        when(self.logger).showView(...).thenReturn()
        sceneView = mock(View2D)
        sceneView.group = ViewGroup.SCENE
        solidView = mock(View2D)
        solidView.group = ViewGroup.SOLIDS
        self.logger.views = [sceneView, solidView]

        self.viewer.show2DAllViews()

        for i in range(2):
            verify(self.logger, times=1).showView(view=None, viewIndex=i, logScale=True, colormap=ANY(str))

    def testWhenShow2DAllViewsWithViewGroup_shouldOnlyShowViewsOfThisGroup(self):
        when(self.logger).showView(...).thenReturn()
        sceneView = mock(View2D)
        sceneView.group = ViewGroup.SCENE
        solidView = mock(View2D)
        solidView.group = ViewGroup.SOLIDS
        self.logger.views = [sceneView, solidView]

        self.viewer.show2DAllViews(ViewGroup.SOLIDS)

        verify(self.logger, times=1).showView(view=None, viewIndex=1, logScale=True, colormap=ANY(str))
        verify(self.logger, times=0).showView(view=None, viewIndex=0, logScale=True, colormap=ANY(str))

    def testWhenShow3DWithScene_shouldDisplayScene(self):
        self.viewer.show3D(visibility=Visibility.SCENE)
        verify(self.scene, times=1).addToViewer(...)
        self.mock3DViewer.show.assert_called_once()

    def testWhenShow3DWithSource_shouldDisplaySource(self):
        self.viewer.show3D(visibility=Visibility.SOURCE)
        verify(self.source, times=1).addToViewer(...)
        self.mock3DViewer.show.assert_called_once()

    def testWhenShow3DWithDefaultPointCloud_shouldDisplayPointCloudOfSolidsAndSurfaceLeaving(self):
        aPointCloud = PointCloud(
            solidPoints=np.array([[0.5, 0, 0, 0, 0]]), surfacePoints=np.array([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]])
        )
        with self._mockPointCloud(aPointCloud):
            self.viewer.show3D(visibility=Visibility.POINT_CLOUD)

        self.mock3DViewer.addDataPoints.assert_called()
        addedSolidPoints = self.mock3DViewer.addDataPoints.call_args_list[0][0][0]
        addedSurfacePoints = self.mock3DViewer.addDataPoints.call_args_list[1][0][0]

        self.assertTrue(np.array_equal(addedSolidPoints, aPointCloud.solidPoints))
        self.assertTrue(np.array_equal(addedSurfacePoints, aPointCloud.leavingSurfacePoints))
        self.mock3DViewer.show.assert_called_once()

    def testGivenNoData_whenShow3DWithPointCloud_shouldNotDisplayPointCloud(self):
        aPointCloud = PointCloud()
        with self._mockPointCloud(aPointCloud):
            self.viewer.show3D(visibility=Visibility.POINT_CLOUD)

        self.mock3DViewer.addDataPoints.assert_not_called()
        self.mock3DViewer.show.assert_called_once()

    def testWhenShow3DWithSurfacePointCloud_shouldOnlyDisplaySurfacePoints(self):
        aPointCloud = PointCloud(
            solidPoints=np.array([[0.5, 0, 0, 0, 0]]), surfacePoints=np.array([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]])
        )
        with self._mockPointCloud(aPointCloud):
            self.viewer.show3D(
                visibility=Visibility.POINT_CLOUD, pointCloudStyle=PointCloudStyle(showSolidPoints=False)
            )

        self.mock3DViewer.addDataPoints.assert_called_once()
        self.assertTrue(
            np.array_equal(self.mock3DViewer.addDataPoints.call_args[0][0], aPointCloud.leavingSurfacePoints)
        )
        self.mock3DViewer.show.assert_called_once()

    def testWhenShow3DWithEnteringSurfacePointCloud_shouldOnlyDisplayEnteringSurfacePoints(self):
        aPointCloud = PointCloud(
            solidPoints=np.array([[0.5, 0, 0, 0, 0]]), surfacePoints=np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 1]])
        )
        with self._mockPointCloud(aPointCloud):
            self.viewer.show3D(
                visibility=Visibility.POINT_CLOUD,
                pointCloudStyle=PointCloudStyle(
                    showSolidPoints=False, showSurfacePointsLeaving=False, showSurfacePointsEntering=True
                ),
            )

        self.mock3DViewer.addDataPoints.assert_called_once()
        self.assertTrue(
            np.array_equal(self.mock3DViewer.addDataPoints.call_args[0][0], aPointCloud.enteringSurfacePointsPositive)
        )
        self.mock3DViewer.show.assert_called_once()

    def testWhenShow3DWithViews_shouldAdd2DImageOfTheseViewsInThe3DDisplay(self):
        self._givenLoggerWithXSceneView()
        sceneView = self.logger.views[0]

        self.viewer.show3D(visibility=Visibility.VIEWS)

        self.mock3DViewer.addImage.assert_called_once()
        addedImage = self.mock3DViewer.addImage.call_args[0][0]
        self.assertTrue(np.array_equal(sceneView.getImageDataWithDefaultAlignment(), addedImage))
        displayedPosition = self.mock3DViewer.addImage.call_args[0][4]
        self.assertEqual(-2.1, displayedPosition)
        self.mock3DViewer.show.assert_called_once()

    def testWhenShow3DWithViewsIndexList_shouldAdd2DImageOfTheseViewsInThe3DDisplay(self):
        self._givenLoggerWithXSceneView()
        sceneView = self.logger.views[0]
        theViewIndex = 9
        when(self.logger).getView(theViewIndex).thenReturn(sceneView)

        self.viewer.show3D(visibility=Visibility.VIEWS, viewsVisibility=[theViewIndex])

        self.mock3DViewer.addImage.assert_called_once()
        addedImage = self.mock3DViewer.addImage.call_args[0][0]
        self.assertTrue(np.array_equal(sceneView.getImageDataWithDefaultAlignment(), addedImage))
        displayedPosition = self.mock3DViewer.addImage.call_args[0][4]
        self.assertEqual(-2.1, displayedPosition)
        self.mock3DViewer.show.assert_called_once()

    def testGiven3DLogger_whenShow3DDefault_shouldDisplayEverythingExceptViews(self):
        aPointCloud = PointCloud()

        with self._mockPointCloud(aPointCloud):
            self.viewer.show3D()

        verify(self.source, times=1).addToViewer(...)
        verify(self.scene, times=1).addToViewer(...)
        self.mock3DViewer.addImage.assert_not_called()
        self.mock3DViewer.show.assert_called_once()

    def testGiven2DLogger_whenShow3DDefault_shouldDisplayEverythingExceptPointCloud(self):
        self._givenLoggerWithXSceneView()
        self.logger.has3D = False

        with self._mockPointCloud(PointCloud()):
            self.viewer.show3D()

        verify(self.source, times=1).addToViewer(...)
        verify(self.scene, times=1).addToViewer(...)
        self.mock3DViewer.addImage.assert_called()
        self.mock3DViewer.show.assert_called_once()

    def testGiven2DLogger_whenShow3DWithDefault3DVisibility_shouldWarnAndDisplayDefault2D(self):
        self._givenLoggerWithXSceneView()
        self.logger.has3D = False

        with self._mockPointCloud(PointCloud()):
            with self.assertWarns(UserWarning):
                self.viewer.show3D(visibility=Visibility.DEFAULT_3D)

        verify(self.source, times=1).addToViewer(...)
        verify(self.scene, times=1).addToViewer(...)
        self.mock3DViewer.addImage.assert_called()
        self.mock3DViewer.show.assert_called_once()

    def _givenLoggerWithXSceneView(self):
        sceneView = View2DProjectionX()
        sceneView.setContext(limits3D=[(-2, 2), (-2, 2), (0, 5)], binSize3D=(1, 1, 1))
        testData = np.array([[1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 0, 2], [4, 0, 0, 3]])
        sceneView.extractData(testData)
        self.logger.views = [sceneView]
        when(self.logger).updateView(sceneView).thenReturn()
        when(self.scene).getBoundingBox().thenReturn(BoundingBox([-2, 2], [-2, 2], [0, 5]))

    @staticmethod
    def _mockPointCloud(pointCloud: PointCloud):
        return patch(
            "pytissueoptics.rayscattering.display.viewer.PointCloudFactory.getPointCloud",
            return_value=pointCloud,
        )
