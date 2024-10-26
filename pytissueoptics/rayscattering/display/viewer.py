from enum import Flag
from typing import List, Tuple, Union

import numpy as np
import os

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.rayscattering.energyLogging.pointCloud import PointCloud
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.statistics import Stats
from pytissueoptics.rayscattering.display.utils import Direction
from pytissueoptics.rayscattering.display.views import ViewGroup, View2D
from pytissueoptics.rayscattering.display.profiles import ProfileFactory
from pytissueoptics.scene import MAYAVI_AVAILABLE, MayaviViewer, ViewPointStyle


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
    def __init__(self, scene: ScatteringScene, source: Source, logger: EnergyLogger):
        assert isinstance(logger, EnergyLogger), "The viewer requires an EnergyLogger."
        self._scene = scene
        self._source = source
        self._logger = logger

        self._viewer3D = None
        self._pointCloudFactory = PointCloudFactory(logger)
        self._profileFactory = ProfileFactory(scene, logger)

    def listViews(self):
        return self._logger.listViews()

    def show3D(self, visibility=Visibility.AUTO, viewsVisibility: Union[ViewGroup, List[int]] = ViewGroup.SCENE,
               pointCloudStyle=PointCloudStyle(), viewsSolidLabels: List[str] = None, viewsSurfaceLabels: List[str] = None,
               viewsLogScale: bool = True, viewsColormap: str = "viridis"):
        if not MAYAVI_AVAILABLE or os.environ.get('PYTISSUE_NO3DDISPLAY','0') == '1':
            utils.warn("Package 'mayavi' is not available. Please install it to use 3D visualizations.")
            return

        self._viewer3D = MayaviViewer(viewPointStyle=ViewPointStyle.OPTICS)

        if visibility == Visibility.AUTO:
            visibility = Visibility.DEFAULT_3D if self._logger.has3D else Visibility.DEFAULT_2D

        if Visibility.DEFAULT_3D in visibility and not self._logger.has3D:
            utils.warn("WARNING: Cannot show3D with Visibility.DEFAULT_3D when no 3D data is available.")
            visibility = Visibility.DEFAULT_2D

        if Visibility.SCENE in visibility:
            self._scene.addToViewer(self._viewer3D)

        if Visibility.SOURCE in visibility:
            self._source.addToViewer(self._viewer3D)

        if Visibility.POINT_CLOUD in visibility:
            self._addPointCloud(pointCloudStyle)

        if Visibility.VIEWS in visibility:
            self._addViews(viewsVisibility, viewsSolidLabels, viewsSurfaceLabels, viewsLogScale, viewsColormap)

        self._viewer3D.show()

    def show3DVolumeSlicer(self, binSize: float = None, logScale: bool = True, interpolate: bool = False,
                           limits: Tuple[tuple, tuple, tuple]=None):
        if not MAYAVI_AVAILABLE:
            utils.warn("ERROR: Package 'mayavi' is not available. Please install it to use 3D visualizations.")
            return

        if not self._logger.has3D:
            utils.warn("ERROR: Cannot show 3D volume slicer without 3D data.")
            return

        if binSize is None:
            binSize = self._logger.defaultBinSize

        limits = limits or self._sceneLimits
        bins = [int((d[1] - d[0]) / binSize) for d in limits]

        # np.histogramdd only works in float64 and requires around 3 times the final memory.
        requiredMemoryInGB = 3 * 8 * bins[0] * bins[1] * bins[2] / 1024**3
        if requiredMemoryInGB > 4:
            utils.warn(f"WARNING: The volume slicer will require a lot of memory ({round(requiredMemoryInGB, 2)} GB). "
                       f"Consider using a larger binSize or tighter limits.")

        points = self._pointCloudFactory.getPointCloudOfSolids().solidPoints
        try:
            hist, _ = np.histogramdd(points[:, 1:], bins=bins, weights=points[:, 0], range=limits)
        except MemoryError:
            utils.warn("ERROR: Not enough memory to create the volume slicer. "
                       "Consider using a larger binSize or tighter limits.")
            return
        hist = hist.astype(np.float32)

        if logScale:
            hist = utils.logNorm(hist)

        from pytissueoptics.rayscattering.display.utils.volumeSlicer import VolumeSlicer
        slicer = VolumeSlicer(hist, interpolate=interpolate)
        slicer.show()

    def show2D(self, view: View2D = None, viewIndex: int = None, logScale: bool = True, colormap: str = "viridis"):
        self._logger.showView(view=view, viewIndex=viewIndex, logScale=logScale, colormap=colormap)

    def show2DAllViews(self, viewGroup=ViewGroup.ALL):
        for i in range(len(self._logger.views)):
            if self._logger.views[i].group not in viewGroup:
                continue
            self.show2D(viewIndex=i)

    def show1D(self, along: Direction, logScale: bool = True,
               solidLabel: str = None, surfaceLabel: str = None, surfaceEnergyLeaving: bool = True,
               limits: Tuple[float, float] = None, binSize: float = None):
        profile = self._profileFactory.create(along, solidLabel, surfaceLabel, surfaceEnergyLeaving, limits, binSize)
        profile.show(logScale=logScale)

    def reportStats(self, solidLabel: str = None, saveToFile: str = None, verbose=True):
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

    def _addViews(self, viewsVisibility: Union[ViewGroup, List[int]], solidLabels: List[str] = None,
                  surfaceLabels: List[str] = None, logScale: bool = True, colormap: str = "viridis"):
        if isinstance(viewsVisibility, list):
            for viewIndex in viewsVisibility:
                self._addView(self._logger.getView(viewIndex), logScale, colormap)
            return

        for view in self._logger.views:
            correctGroup = view.group in viewsVisibility
            correctSolidLabel = solidLabels is None or utils.labelContained(view.solidLabel, solidLabels)
            correctSurfaceLabel = surfaceLabels is None or view.surfaceLabel is None or \
                                  utils.labelContained(view.surfaceLabel, surfaceLabels)
            if correctGroup and correctSolidLabel and correctSurfaceLabel:
                self._addView(view, logScale, colormap)

    def _addView(self, view: View2D, logScale: bool = True, colormap: str = "viridis"):
        self._logger.updateView(view)

        limits = self._sceneLimits
        if view.solidLabel:
            # fixme (limitation): if the solidLabel represents an internal layer of a stack, the limits returned will be
            #  the limits of the whole stack. Could be solved using the surfaceLabels of the layer to compute bbox from
            #  surface polygons. The behaviour is still fine for now and might be what we ultimately want.
            limits = self._scene.getSolid(view.solidLabel).getBoundingBox().xyzLimits

        if view.displayPosition is None:
            viewAxisLimits = sorted(limits[view.axis])
            positionMin, positionMax = viewAxisLimits
            viewSpacing = 0.1

            if view.projectionDirection.isPositive:
                view.displayPosition = positionMin - viewSpacing
            else:
                view.displayPosition = positionMax + viewSpacing

        alignedImage = view.getImageDataWithDefaultAlignment(logScale=logScale)

        alignedCorner = (min(view.limitsU), min(view.limitsV))
        alignedSize = view.size
        if view.axisU > view.axisV:
            alignedCorner = alignedCorner[::-1]
            alignedSize = alignedSize[::-1]

        try:
            self._viewer3D.addImage(alignedImage, alignedSize, alignedCorner, view.axis, view.displayPosition, colormap)
        except MemoryError:
            utils.warn(f"ERROR: Not enough memory to display the view ({view.name}). Consider using a larger bin size.")

    @property
    def _sceneLimits(self):
        sceneBoundingBox = self._scene.getBoundingBox()
        if sceneBoundingBox:
            return sceneBoundingBox.xyzLimits
        else:
            return self._logger.infiniteLimits
