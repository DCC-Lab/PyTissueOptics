import sys
import unittest
from unittest.mock import patch

import numpy as np
from mockito import mock, when, verify, ANY

from pytissueoptics import Direction, View2DProjectionX, ViewGroup
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.display.viewer import Viewer, Visibility, PointCloudStyle
from rayscattering.display.profiles import ProfileFactory, Profile1D
from rayscattering.display.views import View2D
from rayscattering.energyLogging import PointCloudFactory, PointCloud


def patchMayaviRender(func):
    for module in ['show', 'gcf', 'figure', 'clf', 'triangular_mesh', 'points3d']:
        func = patch('mayavi.mlab.' + module)(func)
    return func


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

    def testGivenViewerWithBaseLogger_shouldRaiseException(self):
        with self.assertRaises(AssertionError):
            self.viewer = Viewer(self.scene, self.source, Logger())

    def testWhenListViews_shouldListTheLoggerViews(self):
        when(self.logger).listViews().thenReturn()
        self.viewer.listViews()
        verify(self.logger, times=1).listViews()

    @patch('pytissueoptics.rayscattering.statistics.statistics.Stats.report')
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

        verify(self.logger, times=1).showView(view=view, viewIndex=None,
                                              logScale=True, colormap=ANY(str))

    def testWhenShow2DWithIndex_shouldShow2DViewFromLogger(self):
        when(self.logger).showView(...).thenReturn()

        self.viewer.show2D(viewIndex=1)

        verify(self.logger, times=1).showView(view=None, viewIndex=1,
                                              logScale=True, colormap=ANY(str))

    def testWhenShow2DAllViews_shouldShowAllViews(self):
        when(self.logger).showView(...).thenReturn()
        sceneView = mock(View2D)
        sceneView.group = ViewGroup.SCENE
        solidView = mock(View2D)
        solidView.group = ViewGroup.SOLIDS
        self.logger.views = [sceneView, solidView]

        self.viewer.show2DAllViews()

        for i in range(2):
            verify(self.logger, times=1).showView(view=None, viewIndex=i,
                                                  logScale=True, colormap=ANY(str))

    def testWhenShow2DAllViewsWithViewGroup_shouldOnlyShowViewsOfThisGroup(self):
        when(self.logger).showView(...).thenReturn()
        sceneView = mock(View2D)
        sceneView.group = ViewGroup.SCENE
        solidView = mock(View2D)
        solidView.group = ViewGroup.SOLIDS
        self.logger.views = [sceneView, solidView]

        self.viewer.show2DAllViews(ViewGroup.SOLIDS)

        verify(self.logger, times=1).showView(view=None, viewIndex=1,
                                              logScale=True, colormap=ANY(str))
        verify(self.logger, times=0).showView(view=None, viewIndex=0,
                                              logScale=True, colormap=ANY(str))

    def testWhenShow3DWithoutMayaviInstalled_shouldWarnAndIgnore(self):
        from pytissueoptics.rayscattering.display import viewer
        viewer.MAYAVI_AVAILABLE = False
        with self.assertWarns(UserWarning):
            self.viewer.show3D()

    @patchMayaviRender
    def testWhenShow3DWithScene_shouldDisplayScene(self, mockShow, *args):
        self.viewer.show3D(visibility=Visibility.SCENE)

        verify(self.scene, times=1).addToViewer(...)
        mockShow.assert_called_once()

    @patchMayaviRender
    def testWhenShow3DWithSource_shouldDisplaySource(self, mockShow, *args):
        self.viewer.show3D(visibility=Visibility.SOURCE)

        verify(self.source, times=1).addToViewer(...)
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addDataPoints')
    def testWhenShow3DWithDefaultPointCloud_shouldDisplayPointCloudOfSolidsAndSurfaceLeaving(self, mockAddDataPoints, mockShow, *args):
        mockPointCloudFactory = mock(PointCloudFactory)
        aPointCloud = PointCloud(solidPoints=np.array([[0.5, 0, 0, 0]]),
                                 surfacePoints=np.array([[1, 0, 0, 0], [-1, 0, 0, 0]]))
        when(mockPointCloudFactory).getPointCloud(...).thenReturn(aPointCloud)
        self.viewer._pointCloudFactory = mockPointCloudFactory

        self.viewer.show3D(visibility=Visibility.POINT_CLOUD)

        mockAddDataPoints.assert_called()
        addedSolidPoints = mockAddDataPoints.call_args_list[0].args[0]
        addedSurfacePoints = mockAddDataPoints.call_args_list[1].args[0]
        self.assertTrue(np.array_equal(addedSolidPoints, aPointCloud.solidPoints))
        self.assertTrue(np.array_equal(addedSurfacePoints, aPointCloud.leavingSurfacePoints))
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addDataPoints')
    def testGivenNoData_whenShow3DWithPointCloud_shouldNotDisplayPointCloud(self, mockAddDataPoints, mockShow, *args):
        mockPointCloudFactory = mock(PointCloudFactory)
        aPointCloud = PointCloud()
        when(mockPointCloudFactory).getPointCloud(...).thenReturn(aPointCloud)
        self.viewer._pointCloudFactory = mockPointCloudFactory

        self.viewer.show3D(visibility=Visibility.POINT_CLOUD)

        mockAddDataPoints.assert_not_called()
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addDataPoints')
    def testWhenShow3DWithSurfacePointCloud_shouldOnlyDisplaySurfacePoints(self, mockAddDataPoints, mockShow, *args):
        mockPointCloudFactory = mock(PointCloudFactory)
        aPointCloud = PointCloud(solidPoints=np.array([[0.5, 0, 0, 0]]),
                                 surfacePoints=np.array([[1, 0, 0, 0], [-1, 0, 0, 0]]))
        when(mockPointCloudFactory).getPointCloud(...).thenReturn(aPointCloud)
        self.viewer._pointCloudFactory = mockPointCloudFactory

        self.viewer.show3D(visibility=Visibility.POINT_CLOUD,
                           pointCloudStyle=PointCloudStyle(showSolidPoints=False))

        mockAddDataPoints.assert_called_once()
        self.assertTrue(np.array_equal(mockAddDataPoints.call_args.args[0], aPointCloud.leavingSurfacePoints))
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addDataPoints')
    def testWhenShow3DWithEnteringSurfacePointCloud_shouldOnlyDisplayEnteringSurfacePoints(self, mockAddDataPoints, mockShow, *args):
        mockPointCloudFactory = mock(PointCloudFactory)
        aPointCloud = PointCloud(solidPoints=np.array([[0.5, 0, 0, 0]]),
                                 surfacePoints=np.array([[1, 0, 0, 0], [-1, 1, 1, 1]]))
        when(mockPointCloudFactory).getPointCloud(...).thenReturn(aPointCloud)
        self.viewer._pointCloudFactory = mockPointCloudFactory

        self.viewer.show3D(visibility=Visibility.POINT_CLOUD,
                           pointCloudStyle=PointCloudStyle(showSolidPoints=False,
                                                           showSurfacePointsLeaving=False,
                                                           showSurfacePointsEntering=True))

        mockAddDataPoints.assert_called_once()
        self.assertTrue(np.array_equal(mockAddDataPoints.call_args.args[0], aPointCloud.enteringSurfacePointsPositive))
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addImage')
    def testWhenShow3DWithViews_shouldAdd2DImageOfTheseViewsInThe3DDisplay(self, mockAddImage, mockShow, *args):
        self._givenLoggerWithXSceneView()
        sceneView = self.logger.views[0]

        self.viewer.show3D(visibility=Visibility.VIEWS)

        mockAddImage.assert_called_once()
        addedImage = mockAddImage.call_args.args[0]
        self.assertTrue(np.array_equal(sceneView.getImageDataWithDefaultAlignment(), addedImage))
        displayedPosition = mockAddImage.call_args.args[4]
        self.assertEqual(-2.1, displayedPosition)
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addImage')
    def testWhenShow3DWithViewsIndexList_shouldAdd2DImageOfTheseViewsInThe3DDisplay(self, mockAddImage, mockShow, *args):
        self._givenLoggerWithXSceneView()
        sceneView = self.logger.views[0]
        theViewIndex = 9
        when(self.logger).getView(theViewIndex).thenReturn(sceneView)

        self.viewer.show3D(visibility=Visibility.VIEWS, viewsVisibility=[theViewIndex])

        mockAddImage.assert_called_once()
        addedImage = mockAddImage.call_args.args[0]
        self.assertTrue(np.array_equal(sceneView.getImageDataWithDefaultAlignment(), addedImage))
        displayedPosition = mockAddImage.call_args.args[4]
        self.assertEqual(-2.1, displayedPosition)
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addImage')
    def testGiven3DLogger_whenShow3DDefault_shouldDisplayEverythingExceptViews(self, mockAddImage, mockShow, *args):
        mockPointCloudFactory = mock(PointCloudFactory)
        aPointCloud = PointCloud()
        when(mockPointCloudFactory).getPointCloud(...).thenReturn(aPointCloud)
        self.viewer._pointCloudFactory = mockPointCloudFactory

        self.viewer.show3D()

        verify(self.source, times=1).addToViewer(...)
        verify(self.scene, times=1).addToViewer(...)
        verify(mockPointCloudFactory, times=1).getPointCloud(...)
        mockAddImage.assert_not_called()
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addImage')
    def testGiven2DLogger_whenShow3DDefault_shouldDisplayEverythingExceptPointCloud(self, mockAddImage, mockShow, *args):
        self._givenLoggerWithXSceneView()
        self.logger.has3D = False

        mockPointCloudFactory = mock(PointCloudFactory)
        when(mockPointCloudFactory).getPointCloud(...).thenReturn()
        self.viewer._pointCloudFactory = mockPointCloudFactory

        self.viewer.show3D()

        verify(self.source, times=1).addToViewer(...)
        verify(self.scene, times=1).addToViewer(...)
        verify(mockPointCloudFactory, times=0).getPointCloud(...)
        mockAddImage.assert_called()
        mockShow.assert_called_once()

    @patchMayaviRender
    @patch('pytissueoptics.scene.viewer.MayaviViewer.addImage')
    def testGiven2DLogger_whenShow3DWithDefault3DVisibility_shouldWarnAndDisplayDefault2D(self, mockAddImage, mockShow, *args):
        self._givenLoggerWithXSceneView()
        self.logger.has3D = False

        mockPointCloudFactory = mock(PointCloudFactory)
        when(mockPointCloudFactory).getPointCloud(...).thenReturn()
        self.viewer._pointCloudFactory = mockPointCloudFactory

        with self.assertWarns(UserWarning):
            self.viewer.show3D(visibility=Visibility.DEFAULT_3D)

        verify(self.source, times=1).addToViewer(...)
        verify(self.scene, times=1).addToViewer(...)
        verify(mockPointCloudFactory, times=0).getPointCloud(...)
        mockAddImage.assert_called()
        mockShow.assert_called_once()

    def _givenLoggerWithXSceneView(self):
        sceneView = View2DProjectionX()
        sceneView.setContext(limits3D=[(-2, 2), (-2, 2), (0, 5)], binSize3D=(1, 1, 1))
        testData = np.array([[1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 0, 2], [4, 0, 0, 3]])
        sceneView.extractData(testData)
        self.logger.views = [sceneView]
        when(self.logger).updateView(sceneView).thenReturn()
        when(self.scene).getBoundingBox().thenReturn(BoundingBox([-2, 2], [-2, 2], [0, 5]))
