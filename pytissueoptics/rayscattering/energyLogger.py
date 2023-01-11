from enum import Flag
from typing import Union, List

import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.views import ViewGroup, View2D, ViewFactory
from pytissueoptics.scene.logger.logger import Logger, InteractionKey


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


# todo: in the viewer, set default args: visibility = Visibility.DEFAULT, viewsVisibility: ViewGroup = ViewGroup.SCENE
#  set visibility to DEFAULT_3D if 3D data present, else DEFAULT_2D.


class EnergyLogger(Logger):
    def __init__(self, scene: RayScatteringScene, filepath: str = None, keep3D: bool = True,
                 defaultBinSize: float = 0.01, views: Union[ViewGroup, List[View2D]] = ViewGroup.ALL):
        """
        Log the energy deposited by scattering photons as well as the energy that crossed surfaces. Every interaction
        is linked to a specific solid and surface of the scene when applicable. This `EnergyLogger` has to be given to
        a `Source` instance when propagating.

        :param scene: The scene is required to auto-initialize the 2D views correctly. Even if `keep3D` is True, the
                scene may be used later to correctly create and display the 2D views.
        :param filepath: (Optional) Load the logger from the given file. When saving, the file will be overwritten.
        :param keep3D: (Default to True) If False, logged datapoints are automatically binned to predefined 2D `views`
                and the 3D data is discarded. This allows for a lightweight logger alternative. If True, the 3D data is
                kept and the 2D views are only computed later if displayed.
        :param defaultBinSize: (Default to 0.01) The default bin size to use when binning the 3D data to 2D views.
                In the same physical units as the scene. Custom bin sizes can be specified in each View2D.
        :param views: The 2D views to track, particularly used when `keep3D` is set to False. Can be a ViewGroup flag
                or a list of View2D. Default to ViewGroup.ALL which includes the 3 default XYZ projections for the
                whole scene as well as for each solid, and a projection of each surface in the direction of the surface
                normal for both the energy entering and leaving the surface. When `keep3D` is False, ViewGroup.All is
                required to allow calculation of the detailed report for absorbance and transmittance. When `keep3D` is
                True, the 2D views data is only computed when displayed, so it is equivalent to setting `views` to None.
        """
        self._scene = scene
        self._keep3D = keep3D
        self._defaultBinSize = defaultBinSize

        self._viewFactory = ViewFactory(scene, defaultBinSize)
        self._views = self._viewFactory.build(views)
        # todo: could extract struct of scene and solid limits to avoid having to pass the scene to the ViewFactory
        # todo: could reverse Source->Logger dependence. Source could be passed to Logger with Observer pattern.
        #  Then >>> scene = ...
        #       >>> source = ...
        #       >>> logger = EnergyLogger(source, scene)
        #       >>> source.propagate(scene)
        self._nDataPointsRemoved = 0
        super().__init__(fromFilepath=filepath)  # todo: rewrite save/load to store views and _nDataPointsRemoved

    def addView(self, view: Union[View2D, ViewGroup]):
        self._views += self._viewFactory.build(view)

    @property
    def views(self):
        return self._views

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        """
        Used internally by `Source` when propagating photons. Overwrites the `Logger` method to automatically bin the
        data to 2D views if 3D data is being discarded.
        """
        super().logDataPointArray(array, key)

        if not self._keep3D:
            self._compile()

    def _compile(self):
        for key, data in self._data.items():
            # todo: implement logging of solid-specific views
            if key.surfaceLabel is not None:
                # todo: implement logging of surface views
                continue
            datapoints = data.dataPoints
            if datapoints is None or len(datapoints) == 0:
                continue
            for view in self._views:
                view.extractData(datapoints.array)

        self._nDataPointsRemoved += super().nDataPoints
        self._data.clear()

    def save(self, filepath: str = None):
        raise NotImplementedError()

    def load(self, filepath: str):
        raise NotImplementedError()

    @property
    def nDataPoints(self) -> int:
        """
        Overwrites the `Logger` method to return the total number of data points logged, including the ones discarded.
        """
        if self._keep3D:
            return super().nDataPoints
        else:
            return self._nDataPointsRemoved
