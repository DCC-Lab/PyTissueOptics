import copy
from enum import Enum
from typing import Union, Tuple

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics.scene.logger.logger import Logger, InteractionKey


class Direction(Enum):
    X_POS = 0
    Y_POS = 1
    Z_POS = 2
    X_NEG = 3
    Y_NEG = 4
    Z_NEG = 5

    def sameAxis(self, other) -> bool:
        return self.value % 3 == other.value % 3

    @property
    def axis(self) -> int:
        return self.value % 3

    @property
    def isNegative(self) -> bool:
        return self.value >= 3

    @property
    def isPositive(self) -> bool:
        return not self.isNegative

    @property
    def sign(self) -> int:
        return 1 if self.isPositive else -1


class View2D:
    def __init__(self, projectionDirection: Direction, horizontalDirection: Direction,
                 limits3D: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 bins3D: Union[int, Tuple[int, int, int]], position: float = None, thickness: float = None):
        """
        # fixme: deprecated docstring
        If position is None, the view is a projection (average) of the 3D datapoints on the plane defined by the axis.
        If position is not None, the view is a slice with thickness of the 3D datapoints at the given position.
        The limits and the number of bins are given for the two axes orthogonal to the view axis.
        The limits are (lower, upper) bounds in the same physical units than the data points logged.
        """
        self._projectionDirection = projectionDirection
        self._horizontalDirection = horizontalDirection
        assert not self._projectionDirection.sameAxis(self._horizontalDirection), "Projection and horizontal " \
                                                                                  "directions must be orthogonal."

        bins3D = bins3D if isinstance(bins3D, tuple) else (bins3D, bins3D, bins3D)

        self._limitsU = limits3D[self.axisU]
        self._limitsV = limits3D[self.axisV]
        self._binsU = bins3D[self.axisU]
        self._binsV = bins3D[self.axisV]

        self._dataVU = np.zeros((self._binsV, self._binsU), dtype=np.float32)

        if position is not None or thickness is not None:
            raise NotImplementedError("Slices are not implemented yet.")

    @property
    def axis(self) -> int:
        return self._projectionDirection.axis

    @property
    def axisU(self) -> int:
        return self._horizontalDirection.axis

    @property
    def axisV(self) -> int:
        return 3 - self.axis - self.axisU

    def extractData(self, dataPoints: np.ndarray):
        """
        Data points are (n, 4) arrays with (value, x, y, z).
        """
        u, v, w = dataPoints[:, 1 + self.axisU], dataPoints[:, 1 + self.axisV], dataPoints[:, 0]
        tempData = np.histogram2d(v, u, weights=w, normed=True,
                                  bins=(self._binsV, self._binsU), range=(self._limitsV, self._limitsU))[0]
        self._dataVU += np.flip(tempData, axis=0)

    @property
    def _verticalIsNegative(self):
        requiredAxisForFlip = (self._projectionDirection.axis + self._projectionDirection.sign) % 3
        positiveRequiresFlip = self._horizontalDirection.axis == requiredAxisForFlip
        if self._horizontalDirection.isPositive and positiveRequiresFlip:
            return True
        elif self._horizontalDirection.isNegative and not positiveRequiresFlip:
            return True
        else:
            return False

    def show(self, logScale: bool = True, colormap: str = 'viridis'):
        norm = matplotlib.colors.LogNorm() if logScale else None
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        if self._verticalIsNegative:
            self._dataVU = np.flip(self._dataVU, axis=0)
            self._limitsV = self._limitsV[::-1]

        if self._horizontalDirection.isNegative:
            self._dataVU = np.flip(self._dataVU, axis=1)
            self._limitsU = self._limitsU[::-1]

        plt.imshow(self._dataVU, norm=norm, cmap=cmap, extent=self._limitsU + self._limitsV)
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
        self._views.append(View2D(Direction.X_POS, Direction.Z_POS, self._limits, self._bins))
        self._views.append(View2D(Direction.Y_NEG, Direction.Z_POS, self._limits, self._bins))
        self._views.append(View2D(Direction.Z_POS, Direction.X_NEG, self._limits, self._bins))

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
