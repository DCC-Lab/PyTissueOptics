from enum import Flag

from pytissueoptics.rayscattering.opencl import warnings
from pytissueoptics.rayscattering.pointCloud import PointCloudFactory

from pytissueoptics.rayscattering.views import ViewGroup, View2D
from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.stats import Stats
from pytissueoptics.rayscattering.energyLogger import EnergyLogger


class Visibility(Flag):
    """
    A Visibility is a bit Flag representing what to show inside a 3D visualization. They can be combined with the `|`
    operator (bitwise OR). `DEFAULT` will automatically switch to DEFAULT_3D if 3D data is present, else DEFAULT_2D.
    """
    SCENE = 1
    SOURCE = 2
    VIEWS = 4
    POINT_CLOUD = 8
    POINTS_SURFACES_ENTERING = 16
    POINTS_SURFACES_LEAVING = 32
    POINTS_SURFACES = POINTS_SURFACES_ENTERING | POINTS_SURFACES_LEAVING

    DEFAULT_3D = SCENE | SOURCE | POINT_CLOUD | POINTS_SURFACES_ENTERING
    DEFAULT_2D = SCENE | SOURCE | VIEWS
    DEFAULT = 0


class PointCloudStyle:
    """
    3D display configuration for solid and surface point clouds.

    Visibility attributes:
        solidLabel (Optional[str]): Only show the point cloud specific to a single solid.
        surfaceLabel (Optional[str]): Only show the point cloud specific to a single surface of the solid.
        showSolidPoints (bool): Show the point clouds of the solids.
        showSurfacePoints (bool): Show the point clouds of the surfaces.

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
    def __init__(self, solidLabel: str = None, surfaceLabel: str = None,
                 showSolidPoints: bool = True, showSurfacePoints: bool = True,
                 showPointsAsSpheres: bool = False, pointSize: float = 0.15,
                 scaleWithValue: bool = True, colormap: str = "rainbow", reverseColormap: bool = False,
                 surfacePointSize: float = 0.01, surfaceScaleWithValue: bool = False, surfaceColormap: str = None,
                 surfaceReverseColormap: bool = None):
        self.solidLabel = solidLabel
        self.surfaceLabel = surfaceLabel
        self.showSolidPoints = showSolidPoints
        self.showSurfacePoints = showSurfacePoints
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

    def show3D(self, visibility = Visibility.DEFAULT, viewsVisibility: ViewGroup = ViewGroup.SCENE,
               pointCloudStyle = PointCloudStyle(), sourceSize: float = 0.1):
        from pytissueoptics.scene import MayaviViewer, MAYAVI_AVAILABLE
        if not MAYAVI_AVAILABLE:
            warnings.warn("Package 'mayavi' is not available. Please install it to use 3D visualizations.")
            return

        self._viewer3D = MayaviViewer()

        if visibility == Visibility.DEFAULT:
            visibility = Visibility.DEFAULT_3D if self._logger.has3D else Visibility.DEFAULT_2D

        if Visibility.SCENE in visibility:
            self._scene.addToViewer(self._viewer3D)

        if Visibility.SOURCE in visibility:
            self._source.addToViewer(self._viewer3D, size=sourceSize)

        if Visibility.POINT_CLOUD in visibility:
            self._addPointCloud(pointCloudStyle)

        self._viewer3D.show()

    def _addPointCloud(self, style: PointCloudStyle):
        pointCloud = self._pointCloudFactory.getPointCloud(solidLabel=style.solidLabel, surfaceLabel=style.surfaceLabel)

        if pointCloud.solidPoints is not None:
            self._viewer3D.addDataPoints(pointCloud.solidPoints, scale=style.pointSize,
                                 scaleWithValue=style.scaleWithValue, colormap=style.colormap,
                                 reverseColormap=style.reverseColormap, asSpheres=style.showPointsAsSpheres)
        if pointCloud.surfacePoints is not None:
            self._viewer3D.addDataPoints(pointCloud.leavingSurfacePoints, scale=style.surfacePointSize,
                                 scaleWithValue=style.surfaceScaleWithValue, colormap=style.surfaceColormap,
                                 reverseColormap=style.surfaceReverseColormap, asSpheres=style.showPointsAsSpheres)

    def show3DVolumeSlicer(self):
        pass

    def show2D(self, viewIndex: int = None, view: View2D = None):
        self._logger.showView(viewIndex=viewIndex, view=view)

    def showStats(self):
        if not self._logger.has3D:
            # todo: obtain stats from 2D views
            warnings.warn("WARNING: Cannot show stats if 3D data is not kept.")
            return

        # todo: refactor Stats to only need the logger
        return Stats(self._logger, self._source, self._scene).report()

    def _addViews(self):
        pass

