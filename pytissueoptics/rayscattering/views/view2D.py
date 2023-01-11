import copy
from enum import Enum, Flag
from typing import Tuple, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


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


class ViewGroup(Flag):
    """
    A ViewGroup is a bit Flag representing a group of 2D views. They can be combined with the `|` operator (bitwise OR).

    It has two different usages:
        1. Specify which default views to create when initializing an EnergyLogger, particularly when `keep3D` is False.
        2. Specify which views to show during visualization.

    In the first use case, the default views created are three 2D projections of the whole scene (`SCENE`) or of each solid (`SOLIDS`) in every XYZ direction.
     Except for surface groups, where the default views created include a single 2D projection in the direction of the surface normal.
     `SURFACES_ENTERING` specifies the energy that entered the surface (energy direction opposite to the surface normal).
    """
    SCENE = 1
    SOLIDS = 2
    SURFACES_ENTERING = 4
    SURFACES_LEAVING = 8
    SURFACES = SURFACES_ENTERING | SURFACES_LEAVING
    ALL = SCENE | SOLIDS | SURFACES


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
        Used internally by Logger2D when initializing the views. The limits and the number of bins are given for
        the three dimensions. The view will automatically take the required UV bins and limits from the list. The
        limits are (lower, upper) bounds in the same physical units than the logged data points.
        """
        bins3D = bins3D if isinstance(bins3D, tuple) else (bins3D, bins3D, bins3D)
        self._binsU = bins3D[self.axisU]
        self._binsV = bins3D[self.axisV]
        self._limitsU = sorted(limits3D[self.axisU])
        self._limitsV = sorted(limits3D[self.axisV])

        if self._verticalIsNegative:
            self._limitsV = self._limitsV[::-1]

        if self._horizontalDirection.isNegative:
            self._limitsU = self._limitsU[::-1]

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
        Used internally by Logger2D to store 3D datapoints into this 2D view.
        Data points are (n, 4) arrays with (value, x, y, z).
        """
        u, v, w = dataPoints[:, 1 + self.axisU], dataPoints[:, 1 + self.axisV], dataPoints[:, 0]
        meanUVProjection = np.histogram2d(u, v, weights=w, normed=True, bins=(self._binsU, self._binsV),
                                          range=(sorted(self._limitsU), sorted(self._limitsV)))[0]
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

    def getImageData(self, logNorm: bool = True) -> np.ndarray:
        image = self._dataUV
        if self._verticalIsNegative:
            image = np.flip(image, axis=1)
        if self._horizontalDirection.isNegative:
            image = np.flip(image, axis=0)

        if logNorm:
            eps = 10 ** (-6)
            image /= np.max(image)
            image = np.log(image + eps)
            image -= np.min(image)
            image /= np.max(image)
        return image

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """ Image extent [left, right, bottom, top]. """
        return self._limitsU + self._limitsV

    @property
    def size(self) -> Tuple[float, float]:
        uSize = max(self._limitsU) - min(self._limitsU)
        vSize = max(self._limitsV) - min(self._limitsV)
        return uSize, vSize

    @property
    def minCorner(self) -> Tuple[float, float]:
        return min(self._limitsU), min(self._limitsV)

    def show(self, logScale: bool = True, colormap: str = 'viridis'):
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        image = self.getImageData(logNorm=logScale)

        # N.B.: imshow() expects the data array to be (y, x), so we need to transpose the data array.
        plt.imshow(image.T, cmap=cmap, extent=self.extent)
        plt.xlabel('xyz'[self.axisU])
        plt.ylabel('xyz'[self.axisV])
        plt.show()
