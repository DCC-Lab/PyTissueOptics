from typing import Tuple, Union

import numpy as np

from pytissueoptics.rayscattering.views.view2D import View2D, ViewGroup
from pytissueoptics.rayscattering.views.direction import *


class View2DProjection(View2D):
    def __init__(self, projectionDirection: Direction, horizontalDirection: Direction, solidLabel: str = None,
                 limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 binSize: Union[float, Tuple[int, int]] = None):
        super().__init__(projectionDirection, horizontalDirection,
                         solidLabel=solidLabel, limits=limits, binSize=binSize)

    def _filter(self, dataPoints: np.ndarray) -> np.ndarray:
        return dataPoints


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

    def _filter(self, dataPoints: np.ndarray) -> np.ndarray:
        if self._surfaceEnergyLeaving:
            dataPoints = dataPoints[dataPoints[:, 0] > 0]
        else:
            dataPoints = dataPoints[dataPoints[:, 0] < 0]
            dataPoints[:, 0] *= -1
        return dataPoints

    @property
    def group(self) -> ViewGroup:
        return ViewGroup.SURFACES_LEAVING if self._surfaceEnergyLeaving else ViewGroup.SURFACES_ENTERING


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
