from typing import Union, List

import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.views import ViewGroup, View2D, ViewFactory
from pytissueoptics.scene.logger.logger import Logger, InteractionKey


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
        self._views += self._viewFactory.build([view])

    @property
    def views(self):
        return self._views

    @property
    def has3D(self) -> bool:
        return self._keep3D

    @property
    def defaultBinSize(self) -> float:
        return self._defaultBinSize

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        """
        Used internally by `Source` when propagating photons. Overwrites the `Logger` method to automatically bin the
        data to 2D views if 3D data is being discarded.
        """
        super().logDataPointArray(array, key)

        if not self._keep3D:
            self._compileViews(self._views)
            self._delete3DData()

    def _compileViews(self, views: List[View2D]):
        for key, data in self._data.items():
            datapoints = data.dataPoints
            if datapoints is None or len(datapoints) == 0:
                continue
            for view in views:
                if view.solidLabel and view.solidLabel != key.solidLabel:
                    continue
                if view.surfaceLabel and view.surfaceLabel != key.surfaceLabel:
                    continue
                if view.surfaceLabel is None and key.surfaceLabel is not None:
                    continue
                view.extractData(datapoints.array)

    def _delete3DData(self):
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

    def listViews(self):
        print("Available views:")
        for i, view in enumerate(self._views):
            print(f"\t{i}: {view.description}")

    def showView(self, viewIndex: int = None, view: View2D = None,
                 logScale: bool = True, colormap: str = 'viridis'):
        assert viewIndex is not None or view is not None, "Either `viewIndex` or `view` must be specified."

        if viewIndex is None:
            # todo: if view2D exists, display, else try to flip existing view, else create from scratch with 3D data
            raise NotImplementedError("Displaying a view from scratch is not yet implemented.")

        view = self._views[viewIndex]
        if view.hasData:
            view.show(logScale=logScale, colormap=colormap)
        else:
            self._compileViews([view])
            view.show(logScale=logScale, colormap=colormap)
