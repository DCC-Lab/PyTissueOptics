import copy
from dataclasses import dataclass
from enum import Enum, Flag
from typing import Tuple, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering import utils


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
                 solidLabel: str = None, surfaceLabel: str = None, surfaceEnergyLeaving: bool = True,
                 position: float = None, thickness: float = None,
                 limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
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

        If binSizes is None, it uses the default binSizes from the EnergyLogger.
        If limits is None, it uses the scene limits or the solid limits if a solidLabel is specified.
        """
        self._projectionDirection = projectionDirection
        self._horizontalDirection = horizontalDirection
        assert not self._projectionDirection.isSameAxisAs(self._horizontalDirection), "Projection and horizontal " \
                                                                                      "directions must be orthogonal."

        self._solidLabel = solidLabel
        self._surfaceLabel = surfaceLabel
        self._surfaceEnergyLeaving = surfaceEnergyLeaving
        if self._surfaceLabel is not None and self._solidLabel is None:
            self._surfaceLabel = None
            utils.warn("WARNING [View2D]: A surface label was specified without its corresponding solid label. "
                       "Surface label will be ignored.")

        if position is not None or thickness is not None:
            raise NotImplementedError("Slices are not implemented yet.")
        self._position = position
        self._thickness = thickness

        limits = [sorted(l) for l in limits] if limits else [None, None]
        self._limitsU, self._limitsV = limits
        self._binSize = (binSize, binSize) if isinstance(binSize, (int, float)) else binSize
        self._binsU, self._binsV = None, None

        self._dataUV = None
        self._hasData = False

    @property
    def solidLabel(self) -> str:
        return self._solidLabel

    @property
    def surfaceLabel(self) -> str:
        return self._surfaceLabel

    @property
    def isProjection(self) -> bool:
        return self._position is None and not self.isSurface

    @property
    def isSlice(self) -> bool:
        return self._position is not None

    @property
    def isSurface(self) -> bool:
        return self._surfaceLabel is not None

    def setContext(self, limits: Tuple[Tuple[float, float], Tuple[float, float]],
                   binSize: Tuple[float, float]):
        """
        Used internally by ViewFactory when initializing the views. The limits and the bin sizes are given for
        the 2 dimensions in the axis order (U, V) and in the same physical units than the logged data points.
        They are only used if no custom limits or bin sizes are specified in the constructor.
        """
        if self._limitsU is None:
            self._limitsU, self._limitsV = limits
        self._limitsU, self._limitsV = sorted(self._limitsU), sorted(self._limitsV)

        if self._binSize is None:
            self._binSize = binSize

        limits = [self._limitsU, self._limitsV]
        self._binsU, self._binsV = [int((l[1] - l[0]) / b) for l, b in zip(limits, self._binSize)]

        if self._verticalIsNegative:
            self._limitsV = self._limitsV[::-1]

        if self._horizontalDirection.isNegative:
            self._limitsU = self._limitsU[::-1]

        self._dataUV = np.zeros((self._binsU, self._binsV), dtype=np.float32)

    @property
    def name(self) -> str:
        objectLabel = self.solidLabel if self.solidLabel else "Scene"
        if self._surfaceLabel:
            objectLabel += f" surface {self._surfaceLabel}"
            objectLabel += " (leaving)" if self._surfaceEnergyLeaving else " (entering)"

        return f"{self.__class__.__name__} of {objectLabel}"

    @property
    def description(self) -> str:
        return f"{self.name} towards {self._projectionDirection.name} " \
               f"with {self._horizontalDirection.name} horizontal."

    @property
    def group(self) -> ViewGroup:
        if self.isSurface:
            return ViewGroup.SURFACES_LEAVING if self._surfaceEnergyLeaving else ViewGroup.SURFACES_ENTERING
        else:
            return ViewGroup.SOLIDS if self.solidLabel else ViewGroup.SCENE

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

    @property
    def projectionDirection(self) -> Direction:
        return self._projectionDirection

    def extractData(self, dataPoints: np.ndarray):
        """
        Used internally by Logger2D to store 3D datapoints into this 2D view.
        Data points are (n, 4) arrays with (value, x, y, z).
        """
        # todo: implement slices
        if self.isSurface:
            if self._surfaceEnergyLeaving:
                dataPoints = dataPoints[dataPoints[:, 0] >= 0]
            else:
                dataPoints = dataPoints[dataPoints[:, 0] < 0]
                dataPoints[:, 0] *= -1

        u, v, w = dataPoints[:, 1 + self.axisU], dataPoints[:, 1 + self.axisV], dataPoints[:, 0]
        sumUVProjection = np.histogram2d(u, v, weights=w, normed=False, bins=(self._binsU, self._binsV),
                                          range=(sorted(self._limitsU), sorted(self._limitsV)))[0]
        self._dataUV += np.flip(sumUVProjection, axis=1)
        self._hasData = True

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
            image = utils.logNorm(image)
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

    @property
    def hasData(self) -> bool:
        return self._hasData

    def show(self, logScale: bool = True, colormap: str = 'viridis'):
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        image = self.getImageData(logNorm=logScale)

        # N.B.: imshow() expects the data array to be (y, x), so we need to transpose the data array.
        plt.imshow(image.T, cmap=cmap, extent=self.extent)
        plt.title(self.name)
        plt.xlabel('xyz'[self.axisU])
        plt.ylabel('xyz'[self.axisV])
        plt.show()


DEFAULT_X_VIEW_DIRECTIONS = (Direction.X_POS, Direction.Z_POS)
DEFAULT_Y_VIEW_DIRECTIONS = (Direction.Y_NEG, Direction.Z_POS)
DEFAULT_Z_VIEW_DIRECTIONS = (Direction.Z_POS, Direction.X_NEG)


class View2DProjection(View2D):
    def __init__(self, projectionDirection: Direction, horizontalDirection: Direction, solidLabel: str = None,
                 limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(projectionDirection, horizontalDirection,
                         solidLabel=solidLabel, limits=limits, binSize=binSize)

class View2DProjectionX(View2DProjection):
    def __init__(self, solidLabel: str = None, limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(*DEFAULT_X_VIEW_DIRECTIONS, solidLabel=solidLabel, limits=limits, binSize=binSize)

class View2DProjectionY(View2DProjection):
    def __init__(self, solidLabel: str = None, limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(*DEFAULT_Y_VIEW_DIRECTIONS, solidLabel=solidLabel, limits=limits, binSize=binSize)

class View2DProjectionZ(View2DProjection):
    def __init__(self, solidLabel: str = None, limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(*DEFAULT_Z_VIEW_DIRECTIONS, solidLabel=solidLabel, limits=limits, binSize=binSize)

class View2DSurface(View2D):
    def __init__(self, projectionDirection: Direction, horizontalDirection: Direction, solidLabel: str, surfaceLabel: str,
                 surfaceEnergyLeaving: bool = True, limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(projectionDirection, horizontalDirection, solidLabel=solidLabel, surfaceLabel=surfaceLabel,
                         surfaceEnergyLeaving=surfaceEnergyLeaving, limits=limits, binSize=binSize)

class View2DSurfaceX(View2DSurface):
    def __init__(self, solidLabel: str, surfaceLabel: str, surfaceEnergyLeaving: bool = True,
                 limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(*DEFAULT_X_VIEW_DIRECTIONS, solidLabel=solidLabel, surfaceLabel=surfaceLabel,
                         surfaceEnergyLeaving=surfaceEnergyLeaving, limits=limits, binSize=binSize)

class View2DSurfaceY(View2DSurface):
    def __init__(self, solidLabel: str, surfaceLabel: str, surfaceEnergyLeaving: bool = True,
                 limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(*DEFAULT_Y_VIEW_DIRECTIONS, solidLabel=solidLabel, surfaceLabel=surfaceLabel,
                         surfaceEnergyLeaving=surfaceEnergyLeaving, limits=limits, binSize=binSize)

class View2DSurfaceZ(View2DSurface):
    def __init__(self, solidLabel: str, surfaceLabel: str, surfaceEnergyLeaving: bool = True,
                    limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                    binSize: Union[float, Tuple[int, int]] = None):
            super().__init__(*DEFAULT_Z_VIEW_DIRECTIONS, solidLabel=solidLabel, surfaceLabel=surfaceLabel,
                            surfaceEnergyLeaving=surfaceEnergyLeaving, limits=limits, binSize=binSize)
