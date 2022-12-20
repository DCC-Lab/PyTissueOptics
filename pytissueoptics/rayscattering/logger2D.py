import copy
from typing import Union, Tuple

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics.scene.logger.logger import Logger, InteractionKey


class View2D:
    def __init__(self, axis: str, limits: Tuple[Tuple[float, float], Tuple[float, float]],
                 bins: Union[int, Tuple[int, int]], position: float = None, thickness: float = None):
        """
        If position is None, the view is a projection (average) of the 3D datapoints on the plane defined by the axis.
        If position is not None, the view is a slice with thickness of the 3D datapoints at the given position.
        The limits and the number of bins are given for the two axes orthogonal to the view axis.
        The limits are (lower, upper) bounds in the same physical units than the data points logged.
        """
        self._axis = 'xyz'.index(axis)
        self._limits = limits
        self._bins = bins if isinstance(bins, tuple) else (bins, bins)

        self._data = np.zeros((self._bins[0], self._bins[1]), dtype=np.float32)

        if position is not None or thickness is not None:
            raise NotImplementedError("Slices are not implemented yet.")

    def extractData(self, dataPoints: np.ndarray):
        """ Data points are (n, 4) arrays with (value, x, y, z). """
        u, v, w = dataPoints[:, 1 + self.axisU], dataPoints[:, 1 + self.axisV], dataPoints[:, 0]
        self._data += np.histogram2d(u, v, weights=w, bins=self._bins, range=self._limits, normed=True)[0]

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def axisU(self) -> int:
        return (self._axis + 1) % 3

    @property
    def axisV(self) -> int:
        return (self._axis + 2) % 3

    def show(self, logScale: bool = True, colormap: str = 'viridis'):
        norm = matplotlib.colors.LogNorm() if logScale else None
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])
        plt.imshow(self._data, norm=norm, cmap=cmap, extent=self._limits[0] + self._limits[1])
        plt.xlabel('xyz'[self.axisU])
        plt.ylabel('xyz'[self.axisV])
        plt.show()


class Logger2D(Logger):
    """
    Lightweight logger alternative. When datapoints are logged, they are automatically binned to
    predefined 2D views and the 3D data is discarded.
    """
    # todo: overwrite load() and save() to track the 2D views instead of the 3D data points
    # fixme: Only supports DataPoints (n, 4) => Implicit violation of Logger contract (LSP)
    def __init__(self, limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 bins: Union[int, Tuple[int, int, int]]):
        self._limits = limits
        self._bins = bins if isinstance(bins, tuple) else (bins, bins, bins)
        self._views = []

        self._initViews()
        super().__init__()

    def _initViews(self):
        self._views = []
        self._views.append(View2D('x', self._limits[1:], self._bins[1:]))
        self._views.append(View2D('y', (self._limits[0], self._limits[2]), (self._bins[0], self._bins[2])))
        self._views.append(View2D('z', self._limits[:2], self._bins[:2]))

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
