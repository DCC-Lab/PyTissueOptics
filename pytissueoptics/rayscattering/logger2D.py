from enum import Flag
from typing import Union, Tuple, List

import numpy as np

from pytissueoptics import RayScatteringScene
from pytissueoptics.scene.logger.logger import Logger, InteractionKey


class Visibility(Flag):
    """
    A Visibility is a bit Flag representing what to show inside a 3D visualization. They can be combined with the `|` operator (bitwise OR).
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

class Logger2D(Logger):
    """
    Lightweight logger alternative. When datapoints are logged, they are automatically binned to
    predefined 2D views and the 3D data is discarded.
    """

    # todo: overwrite load() and save() to track the 2D views instead of the 3D data points
    # fixme: Only supports DataPoints (n, 4) => Implicit violation of Logger contract (LSP)
    def __init__(self, limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 bins: Union[int, Tuple[int, int, int]], includeDefaultViews: bool = True):
        self._limits = limits
        self._bins = bins if isinstance(bins, tuple) else (bins, bins, bins)
        self._views = []

        if includeDefaultViews:
            self._addDefaultViews()
        super().__init__()

    def _addDefaultViews(self):
        self.addView(View2D(Direction.X_POS, Direction.Z_POS))
        self.addView(View2D(Direction.Y_NEG, Direction.Z_POS))
        self.addView(View2D(Direction.Z_POS, Direction.X_NEG))

    def addView(self, view: View2D):
        view.setContext(self._limits, self._bins)
        self._views.append(view)

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        super().logDataPointArray(array, key)
        self._compile()

    def _compile(self):
        for key, data in self._data.items():
            if key.surfaceLabel is not None:
                # skipping surface data for now
                continue
            datapoints = data.dataPoints
            if datapoints is None or len(datapoints) == 0:
                continue
            for view in self._views:
                view.extractData(datapoints.array)
        self._data.clear()

    @property
    def views(self):
        return self._views
