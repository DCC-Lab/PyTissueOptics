from enum import Flag

from pytissueoptics.rayscattering.opencl import warnings

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
    """ 3D display configuration dataclass for solid and surface point cloud. """
    def __init__(self, sourceSize: float = 0.1, showPointsAsSpheres: bool = True, pointSize: float = 0.15,
                 scaleWithValue: bool = True, colormap: str = "rainbow", reverseColormap: bool = False,
                 surfacePointSize: float = 0.01, surfaceScaleWithValue: bool = False, surfaceColormap: str = None,
                 surfaceReverseColormap: bool = None):
        self.sourceSize = sourceSize
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

    def show3D(self, visibility = Visibility.DEFAULT, viewsVisibility: ViewGroup = ViewGroup.SCENE,
               pointCloudStyle = PointCloudStyle()):
        pass

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

    def _add3DPointCloud(self):
        pass

    def _addViews(self):
        pass


class PointCloudFactory:
    pass
