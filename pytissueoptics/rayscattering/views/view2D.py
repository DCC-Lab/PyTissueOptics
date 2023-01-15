import copy
from enum import Flag
from typing import Tuple, Union, Optional, List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.views.direction import *


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
        self._position = position
        self._thickness = thickness

        limits = [sorted(l) for l in limits] if limits else [None, None]
        self._limitsU, self._limitsV = limits
        self._binSize = (binSize, binSize) if isinstance(binSize, (int, float)) else binSize
        self._binsU, self._binsV = None, None

        self._dataUV = None
        self._hasData = False
        self.displayPosition = position

    def flip(self):
        """ Flips the view as if it was seen from behind. """
        flipHorizontal = self._projectionDirection.axis != 1
        self._projectionDirection = Direction((self._projectionDirection.value + 3) % 6)
        if flipHorizontal:
            self._horizontalDirection = Direction((self._horizontalDirection.value + 3) % 6)

    @property
    def solidLabel(self) -> Optional[str]:
        return self._solidLabel

    @property
    def surfaceLabel(self) -> Optional[str]:
        return self._surfaceLabel

    @property
    def surfaceEnergyLeaving(self) -> bool:
        return self._surfaceEnergyLeaving

    def setContext(self, limits3D: List[Tuple[float, float]], binSize3D: Tuple[float, float, float]):
        """
        Used internally by ViewFactory when initializing the views. The limits and the bin sizes are given for
        all 3 dimensions and in the same physical units than the logged data points. They are only used if no custom
        limits or bin sizes are specified in the constructor.
        """
        if self._limitsU is None:
            self._limitsU, self._limitsV = limits3D[self.axisU], limits3D[self.axisV]
        self._limitsU, self._limitsV = sorted(self._limitsU), sorted(self._limitsV)

        if self._binSize is None:
            self._binSize = (binSize3D[self.axisU], binSize3D[self.axisV])

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
        dataPoints = self._filter(dataPoints)
        if dataPoints.size == 0:
            return

        u, v, w = dataPoints[:, 1 + self.axisU], dataPoints[:, 1 + self.axisV], dataPoints[:, 0]
        sumUVProjection = np.histogram2d(u, v, weights=w, normed=False, bins=(self._binsU, self._binsV),
                                          range=(sorted(self._limitsU), sorted(self._limitsV)))[0]
        self._dataUV += np.flip(sumUVProjection, axis=1)
        self._hasData = True

    def _filter(self, dataPoints: np.ndarray) -> np.ndarray:
        """
        Filters the data points to only keep the ones that are relevant to this view.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

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

    def getImageData(self, logScale: bool = True, autoFlip=True) -> np.ndarray:
        image = self._dataUV
        if logScale and self._hasData:
            image = utils.logNorm(image)
        if not autoFlip:
            return image
        if self._verticalIsNegative:
            image = np.flip(image, axis=1)
        if self._horizontalDirection.isNegative:
            image = np.flip(image, axis=0)
        return image

    def getImageDataWithDefaultAlignment(self, logScale: bool = True) -> np.ndarray:
        """
        Returns the image data aligned with the default 2D alignment for its projection axis. This is used to
        simplify the 3D display of any 2D image. If current view is already aligned with one of the default 2D
        alignments, it will simply return getImageData(). The projection axis (absolute) remains the same.

        The default 2D alignments (projectionDirection, horizontalDirection) are:
            X: (Direction.X_POS, Direction.Z_POS)
            Y: (Direction.Y_NEG, Direction.Z_POS)
            Z: (Direction.Z_POS, Direction.X_NEG)

        Horizontal axis clockwise order for each projection axis starting with the required value:
            X_POS projection: (Z_POS, Y_POS, Z_NEG, Y_NEG) which equals Direction values (2, 1, 5, 4)
            Y_NEG projection: (Z_POS, X_POS, Z_NEG, X_NEG) which equals Direction values (2, 0, 5, 3)
            Z_POS projection: (X_NEG, Y_POS, X_POS, Y_NEG) which equals Direction values (3, 1, 0, 4)
        Using this, the index of the current horizontal equals the number of counter-clockwise 90 degree rotations.

        Correction algorithm:
            If projection is X_NEG, flip horizontally. This will flip horizontalDirection.
            If projection is Y_POS, flip vertically. This will not flip horizontalDirection.
            If projection is Z_NEG, flip horizontally. This will flip horizontalDirection.
            Rotate 90 degrees counter-clockwise until horizontalDirection equals the first value in the list above.
        """
        alignedDirections = [DEFAULT_X_VIEW_DIRECTIONS, DEFAULT_Y_VIEW_DIRECTIONS, DEFAULT_Z_VIEW_DIRECTIONS][self.axis]
        requiredProjection, requiredHorizontal = alignedDirections
        currentProjection, currentHorizontal = self._projectionDirection, self._horizontalDirection

        image = self.getImageData(logScale=logScale)

        if currentProjection != requiredProjection:
            if self.axis == 1:
                image = np.flip(image, axis=1)
            else:
                image = np.flip(image, axis=0)
                currentHorizontal = Direction((currentHorizontal.value + 3) % 6)

        horizontalOrder = {0: [2, 1, 5, 4], 1: [2, 0, 5, 3], 2: [3, 1, 0, 4]}
        horizontalIndex = horizontalOrder[self.axis].index(currentHorizontal.value)
        image = np.rot90(image, horizontalIndex)

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
    def limitsU(self) -> Tuple[float, float]:
        return self._limitsU

    @property
    def limitsV(self) -> Tuple[float, float]:
        return self._limitsV

    @property
    def binsU(self) -> int:
        return self._binsU

    @property
    def binsV(self) -> int:
        return self._binsV

    def show(self, logScale: bool = True, colormap: str = 'viridis'):
        cmap = copy.copy(matplotlib.cm.get_cmap(colormap))
        cmap.set_bad(cmap.colors[0])

        image = self.getImageData(logNorm=logScale)

        # N.B.: imshow() expects the data to be (y, x), so we need to transpose the array.
        plt.imshow(image.T, cmap=cmap, extent=self.extent)
        plt.title(self.name)
        plt.xlabel('xyz'[self.axisU])
        plt.ylabel('xyz'[self.axisV])
        plt.show()

    def isEqualTo(self, other: 'View2D') -> bool:
        if not self.isContainedBy(other):
            return False
        if self._projectionDirection != other._projectionDirection:
            return False
        if self._horizontalDirection != other._horizontalDirection:
            return False
        if (self._binsU, self._binsV) != (other._binsU, other._binsV):
            return False
        limits = sorted(self._limitsU), sorted(self._limitsV)
        otherLimits = sorted(other._limitsU), sorted(other._limitsV)
        if limits != otherLimits:
            return False
        return True

    def isContainedBy(self, other: 'View2D') -> bool:
        if self._projectionDirection.axis != other._projectionDirection.axis:
            return False
        if not utils.labelsEqual(self._solidLabel, other._solidLabel):
            return False
        if not utils.labelsEqual(self._surfaceLabel, other._surfaceLabel):
            return False
        if self._surfaceEnergyLeaving != other._surfaceEnergyLeaving:
            return False
        if self._position != other._position:
            return False
        if self._thickness != other._thickness:
            return False

        # TODO: change/remove the following once the algorithm can extract a view contained inside a bigger view.
        isTransposed = self.axisU != other.axisU
        if isTransposed:
            bins = self._binsV, self._binsU
            limits = sorted(self._limitsV), sorted(self._limitsU)
        else:
            bins = self._binsU, self._binsV
            limits = sorted(self._limitsU), sorted(self._limitsV)
        otherBins = other._binsU, other._binsV
        otherLimits = sorted(other._limitsU), sorted(other._limitsV)

        if bins != otherBins:
            return False
        if limits != otherLimits:
            return False
        return True

    def initDataFrom(self, source: 'View2D'):
        """ Extract data from one view to another when there is only a difference in orientation. """
        assert self.isContainedBy(source), "Cannot extract data from views that are not equivalent."

        dataUV = source._dataUV.copy()
        if source.axisU == self.axisU:
            self._dataUV = dataUV
        else:
            self._dataUV = dataUV.T
        self._hasData = source._hasData

    def getSum(self) -> float:
        return float(np.sum(self._dataUV))
