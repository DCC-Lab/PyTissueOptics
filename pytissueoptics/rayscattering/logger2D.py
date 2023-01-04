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

    def isSameAxisAs(self, other) -> bool:
        return self.value % 3 == other.value % 3

    @property
    def axis(self) -> int:
        """ Returns an integer between 0 and 2 representing the x, y, or z axis, ignoring direction sign. """
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
                 position: float = None, thickness: float = None):
        """
        The 2D view plane is obtained by looking towards the 'projectionDirection'. The 'horizontalDirection'
        represents which axis to use as the horizontal axis in the resulting 2D view. If the 'horizontalDirection' is
        negative, the horizontal axis will go towards negative values. This 'horizontalDirection' sets how the 2D
        view will be oriented when displayed. The vertical direction cannot be set since it is inferred from the
        previous two parameters which yields a single valid direction. For example, with a given 'horizontalDirection',
        flipping the sign of the 'projectionDirection' will flip the resulting vertical axis.

        If position is None, the view is a projection (average) of the 3D datapoints on the plane defined by
        projectionDirection (`thickness` is ignored). If position is not None, the view is a slice of the given
        'thickness' of the 3D datapoints at the given position.
        """
        self._projectionDirection = projectionDirection
        self._horizontalDirection = horizontalDirection
        assert not self._projectionDirection.isSameAxisAs(self._horizontalDirection), "Projection and horizontal " \
                                                                                      "directions must be orthogonal."

        if position is not None or thickness is not None:
            raise NotImplementedError("Slices are not implemented yet.")

        self._limitsU = None
        self._limitsV = None
        self._binsU = None
        self._binsV = None
        
        self._dataUV = None

    def setContext(self, limits3D: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                   bins3D: Union[int, Tuple[int, int, int]]):
        """
        Used internally by the Logger2D when initializing the views. The limits and the number of bins are given for
        the three dimensions. The view will automatically take the required UV bins and limits from the list. The
        limits are (lower, upper) bounds in the same physical units than the logged data points.
        """
        bins3D = bins3D if isinstance(bins3D, tuple) else (bins3D, bins3D, bins3D)
        self._binsU = bins3D[self.axisU]
        self._binsV = bins3D[self.axisV]
        self._limitsU = limits3D[self.axisU]
        self._limitsV = limits3D[self.axisV]

        self._dataUV = np.zeros((self._binsU, self._binsV), dtype=np.float32)

    @property
    def axis(self) -> int:
        """ The axis that represents the plane of the 2D view. """
        return self._projectionDirection.axis

    @property
    def axisU(self) -> int:
        """ The horizontal axis of the 2D view. Could also be referred to as the 'x' axis. """
        return self._horizontalDirection.axis

    @property
    def axisV(self) -> int:
        """ The vertical axis of the 2D view. Could also be referred to as the 'y' axis. """
        return 3 - self.axis - self.axisU

    def extractData(self, dataPoints: np.ndarray):
        """
        Data points are (n, 4) arrays with (value, x, y, z).
        """
        u, v, w = dataPoints[:, 1 + self.axisU], dataPoints[:, 1 + self.axisV], dataPoints[:, 0]
        meanUVProjection = np.histogram2d(u, v, weights=w, normed=True,
                                          bins=(self._binsU, self._binsV), range=(self._limitsU, self._limitsV))[0]
        self._dataUV += np.flip(meanUVProjection, axis=1)

    @property
    def _verticalIsNegative(self) -> bool:
        """ Algorithm for cartesian axes to know if the resulting vertical axis is negative (the axis unit vector
        goes down from the viewer's point of view).
        """
        horizontalAxisForNegativeVertical = (self._projectionDirection.axis + self._projectionDirection.sign) % 3
        verticalIsNegativeWithPositiveHorizontal = self._horizontalDirection.axis == horizontalAxisForNegativeVertical
        if self._horizontalDirection.isPositive:
            return verticalIsNegativeWithPositiveHorizontal
        return not verticalIsNegativeWithPositiveHorizontal

    def show(self, logScale: bool = True, colormap: str = 'viridis'):
        norm = matplotlib.colors.LogNorm() if logScale else None
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        if self._verticalIsNegative:
            self._dataUV = np.flip(self._dataUV, axis=1)
            self._limitsV = self._limitsV[::-1]

        if self._horizontalDirection.isNegative:
            self._dataUV = np.flip(self._dataUV, axis=0)
            self._limitsU = self._limitsU[::-1]

        # N.B.: imshow() expects the data array to be (y, x), so we need to transpose the data array.
        plt.imshow(self._dataUV.T, norm=norm, cmap=cmap, extent=self._limitsU + self._limitsV)
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
