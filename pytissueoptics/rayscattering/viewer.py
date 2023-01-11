from enum import Flag

import numpy as np

from pytissueoptics.rayscattering.energyLogger import EnergyLogger
from pytissueoptics.rayscattering.opencl import warnings
from pytissueoptics.rayscattering.pointCloud import PointCloudFactory, PointCloud
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.statistics import Stats
from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.views import ViewGroup, View2D


class Visibility(Flag):
    """
    A Visibility is a bit Flag representing what to show inside a 3D visualization. They can be combined with the `|`
    operator (bitwise OR). `AUTO` will automatically switch to DEFAULT_3D if 3D data is present, else DEFAULT_2D.
    """
    SCENE = 1
    SOURCE = 2
    POINT_CLOUD = 4
    VIEWS = 8
    DEFAULT_3D = SCENE | SOURCE | POINT_CLOUD
    DEFAULT_2D = SCENE | SOURCE | VIEWS
    AUTO = 0


class PointCloudStyle:
    """
    3D display configuration for solid and surface point clouds.

    Visibility attributes:
        solidLabel (Optional[str]): Only show the point cloud specific to a single solid.
        surfaceLabel (Optional[str]): Only show the point cloud specific to a single surface of the solid.
        showSolidPoints (bool): Show the point clouds of the solids.
        showSurfacePointsLeaving (bool): Show energy that left the surface (direction with surface normal).
        showSurfacePointsEntering (bool): Show energy that entered the surface (direction opposite to surface normal).

    Other attributes:
        showPointsAsSpheres (bool): Show the points as spheres or as dots. Dots require less memory.
        pointSize (float): Reference diameter of the points in the point cloud when drawn as spheres.
        scaleWithValue (bool): Scale the points with their value. A value of 1 yields the `pointSize`.
        colormap (str): The name of the colormap to use for the point cloud.
        reverseColormap (bool): Reverse the colormap.
        surfacePointSize (float): Same as `pointSize`, but for the surface points.
        surfaceScaleWithValue (bool): Same as `scaleWithValue` but for the surface points.
        surfaceColormap (str): Same as `colormap` but for the surface points.
        surfaceReverseColormap (bool): Same as `reverseColormap` but for the surface points.
    """

    def __init__(self, solidLabel: str = None, surfaceLabel: str = None, showSolidPoints: bool = True,
                 showSurfacePointsLeaving: bool = True, showSurfacePointsEntering: bool = False,
                 showPointsAsSpheres: bool = False, pointSize: float = 0.15, scaleWithValue: bool = True,
                 colormap: str = "rainbow", reverseColormap: bool = False, surfacePointSize: float = 0.01,
                 surfaceScaleWithValue: bool = False, surfaceColormap: str = None, surfaceReverseColormap: bool = None):
        self.solidLabel = solidLabel
        self.surfaceLabel = surfaceLabel
        self.showSolidPoints = showSolidPoints
        self.showSurfacePointsLeaving = showSurfacePointsLeaving
        self.showSurfacePointsEntering = showSurfacePointsEntering
        self.showPointsAsSpheres = showPointsAsSpheres

        self.pointSize = pointSize
        self.scaleWithValue = scaleWithValue
        self.colormap = colormap
        self.reverseColormap = reverseColormap

        self.surfacePointSize = surfacePointSize
        self.surfaceScaleWithValue = surfaceScaleWithValue
        self.surfaceColormap = colormap if surfaceColormap is None else surfaceColormap
        self.surfaceReverseColormap = reverseColormap if surfaceReverseColormap is None else surfaceReverseColormap


class Viewer:
    def __init__(self, scene: RayScatteringScene, source: Source, logger: EnergyLogger):
        self._scene = scene
        self._source = source
        self._logger = logger

        self._viewer3D = None
        self._pointCloudFactory = PointCloudFactory(logger)

    def show3D(self, visibility=Visibility.AUTO, viewsVisibility: ViewGroup = ViewGroup.SCENE,
               pointCloudStyle=PointCloudStyle(), sourceSize: float = 0.1):
        from pytissueoptics.scene import MayaviViewer, MAYAVI_AVAILABLE
        if not MAYAVI_AVAILABLE:
            warnings.warn("Package 'mayavi' is not available. Please install it to use 3D visualizations.")
            return

        self._viewer3D = MayaviViewer()

        if visibility == Visibility.AUTO:
            visibility = Visibility.DEFAULT_3D if self._logger.has3D else Visibility.DEFAULT_2D

        if Visibility.SCENE in visibility:
            self._scene.addToViewer(self._viewer3D)

        if Visibility.SOURCE in visibility:
            self._source.addToViewer(self._viewer3D, size=sourceSize)

        if Visibility.POINT_CLOUD in visibility:
            self._addPointCloud(pointCloudStyle)

        if Visibility.VIEWS in visibility:
            self._addViews(viewsVisibility)

        self._viewer3D.show()

    def show3DVolumeSlicer(self):
        pass

    def show2D(self, viewIndex: int = None, view: View2D = None):
        self._logger.showView(viewIndex=viewIndex, view=view)

    def showAllViews(self):
        for i in range(len(self._logger.views)):
            self.show2D(viewIndex=i)

    def listViews(self):
        return self._logger.listViews()

    def reportStats(self, solidLabel: str = None, saveToFile: str = None, verbose=True):
        if not self._logger.has3D:
            # todo: obtain stats from 2D views
            warnings.warn("WARNING: Stats without 3D data is not yet implemented.")
            return

        stats = Stats(self._logger)
        stats.report(solidLabel=solidLabel, saveToFile=saveToFile, verbose=verbose)

    def _addPointCloud(self, style: PointCloudStyle):
        pointCloud = self._pointCloudFactory.getPointCloud(solidLabel=style.solidLabel, surfaceLabel=style.surfaceLabel)

        self._drawPointCloudOfSolids(pointCloud, style)
        self._drawPointCloudOfSurfaces(pointCloud, style)

    def _drawPointCloudOfSolids(self, pointCloud: PointCloud, style: PointCloudStyle):
        if pointCloud.solidPoints is None:
            return
        if not style.showSolidPoints:
            return

        self._viewer3D.addDataPoints(pointCloud.solidPoints, scale=style.pointSize,
                                     scaleWithValue=style.scaleWithValue, colormap=style.colormap,
                                     reverseColormap=style.reverseColormap, asSpheres=style.showPointsAsSpheres)

    def _drawPointCloudOfSurfaces(self, pointCloud: PointCloud, style: PointCloudStyle):
        if pointCloud.surfacePoints is None:
            return

        surfacePoints = np.empty((0, 4))

        if style.showSurfacePointsLeaving:
            surfacePoints = np.vstack((surfacePoints, pointCloud.leavingSurfacePoints))
        if style.showSurfacePointsEntering:
            surfacePoints = np.vstack((surfacePoints, pointCloud.enteringSurfacePointsPositive))
        if len(surfacePoints) == 0:
            return

        self._viewer3D.addDataPoints(surfacePoints, scale=style.surfacePointSize,
                                     scaleWithValue=style.surfaceScaleWithValue, colormap=style.surfaceColormap,
                                     reverseColormap=style.surfaceReverseColormap,
                                     asSpheres=style.showPointsAsSpheres)

    def _addViews(self, viewsVisibility: ViewGroup):
        if viewsVisibility != ViewGroup.SCENE:
            raise NotImplementedError("Only 'ViewGroup.SCENE' can be displayed for now.")

        for view in self._logger.views:
            # todo: assert correct view group
            self._addView(view)

    def _addView(self, view: View2D):
        sceneLimits = self._scene.getBoundingBox().xyzLimits
        viewAxisLimits = sorted(sceneLimits[view.axis])
        positionMin, positionMax = viewAxisLimits
        viewSpacing = 0.1

        if view.projectionDirection.isPositive:
            position = positionMin - viewSpacing
        else:
            position = positionMax + viewSpacing

        self._viewer3D.addImage(view.getImageData(), view.size, view.minCorner, view.axis, position)
