from typing import List, Tuple, Union

import numpy as np

from pytissueoptics.rayscattering.display.utils import (
    DEFAULT_X_VIEW_DIRECTIONS,
    DEFAULT_Y_VIEW_DIRECTIONS,
    DEFAULT_Z_VIEW_DIRECTIONS,
    Direction,
)
from pytissueoptics.rayscattering.display.views.view2D import View2D, ViewGroup
from pytissueoptics.rayscattering.energyLogging.energyType import EnergyType


class View2DProjection(View2D):
    def __init__(
        self,
        projectionDirection: Direction,
        horizontalDirection: Direction,
        solidLabel: str = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            projectionDirection,
            horizontalDirection,
            solidLabel=solidLabel,
            limits=limits,
            binSize=binSize,
            energyType=energyType,
        )

    def _filter(self, dataPoints: np.ndarray) -> np.ndarray:
        return dataPoints


class View2DProjectionX(View2DProjection):
    def __init__(
        self,
        solidLabel: str = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            *DEFAULT_X_VIEW_DIRECTIONS, solidLabel=solidLabel, limits=limits, binSize=binSize, energyType=energyType
        )


class View2DProjectionY(View2DProjection):
    def __init__(
        self,
        solidLabel: str = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            *DEFAULT_Y_VIEW_DIRECTIONS, solidLabel=solidLabel, limits=limits, binSize=binSize, energyType=energyType
        )


class View2DProjectionZ(View2DProjection):
    def __init__(
        self,
        solidLabel: str = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            *DEFAULT_Z_VIEW_DIRECTIONS, solidLabel=solidLabel, limits=limits, binSize=binSize, energyType=energyType
        )


class View2DSurface(View2D):
    def __init__(
        self,
        projectionDirection: Direction,
        horizontalDirection: Direction,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool = True,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
    ):
        super().__init__(
            projectionDirection,
            horizontalDirection,
            solidLabel=solidLabel,
            surfaceLabel=surfaceLabel,
            surfaceEnergyLeaving=surfaceEnergyLeaving,
            limits=limits,
            binSize=binSize,
        )

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
    def __init__(
        self,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool = True,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
    ):
        super().__init__(
            *DEFAULT_X_VIEW_DIRECTIONS,
            solidLabel=solidLabel,
            surfaceLabel=surfaceLabel,
            surfaceEnergyLeaving=surfaceEnergyLeaving,
            limits=limits,
            binSize=binSize,
        )


class View2DSurfaceY(View2DSurface):
    def __init__(
        self,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool = True,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
    ):
        super().__init__(
            *DEFAULT_Y_VIEW_DIRECTIONS,
            solidLabel=solidLabel,
            surfaceLabel=surfaceLabel,
            surfaceEnergyLeaving=surfaceEnergyLeaving,
            limits=limits,
            binSize=binSize,
        )


class View2DSurfaceZ(View2DSurface):
    def __init__(
        self,
        solidLabel: str,
        surfaceLabel: str,
        surfaceEnergyLeaving: bool = True,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
    ):
        super().__init__(
            *DEFAULT_Z_VIEW_DIRECTIONS,
            solidLabel=solidLabel,
            surfaceLabel=surfaceLabel,
            surfaceEnergyLeaving=surfaceEnergyLeaving,
            limits=limits,
            binSize=binSize,
        )


class View2DSlice(View2D):
    def __init__(
        self,
        projectionDirection: Direction,
        horizontalDirection: Direction,
        position: float,
        solidLabel: str = None,
        thickness: float = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            projectionDirection,
            horizontalDirection,
            solidLabel=solidLabel,
            position=position,
            thickness=thickness,
            limits=limits,
            binSize=binSize,
            energyType=energyType,
        )

    def setContext(self, limits3D: List[Tuple[float, float]], binSize3D: Tuple[float, float, float]):
        if self._thickness is None:
            self._thickness = binSize3D[self.axis]
        super().setContext(limits3D, binSize3D)

    def _filter(self, dataPoints: np.ndarray) -> np.ndarray:
        dataPositions = dataPoints[:, 1 + self.axis]
        insideSlice = np.logical_and(
            dataPositions > self._position - self._thickness / 2, dataPositions < self._position + self._thickness / 2
        )
        return dataPoints[insideSlice]


class View2DSliceX(View2DSlice):
    def __init__(
        self,
        position: float,
        solidLabel: str = None,
        thickness: float = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            *DEFAULT_X_VIEW_DIRECTIONS,
            position=position,
            solidLabel=solidLabel,
            thickness=thickness,
            limits=limits,
            binSize=binSize,
            energyType=energyType,
        )


class View2DSliceY(View2DSlice):
    def __init__(
        self,
        position: float,
        solidLabel: str = None,
        thickness: float = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            *DEFAULT_Y_VIEW_DIRECTIONS,
            position=position,
            solidLabel=solidLabel,
            thickness=thickness,
            limits=limits,
            binSize=binSize,
            energyType=energyType,
        )


class View2DSliceZ(View2DSlice):
    def __init__(
        self,
        position: float,
        solidLabel: str = None,
        thickness: float = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
        binSize: Union[float, Tuple[int, int]] = None,
        energyType=EnergyType.DEPOSITION,
    ):
        super().__init__(
            *DEFAULT_Z_VIEW_DIRECTIONS,
            position=position,
            solidLabel=solidLabel,
            thickness=thickness,
            limits=limits,
            binSize=binSize,
            energyType=energyType,
        )
