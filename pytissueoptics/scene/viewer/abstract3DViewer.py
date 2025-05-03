from abc import abstractmethod

import numpy as np

from pytissueoptics.scene.solids import Solid

from .viewPoint import ViewPointStyle


class Abstract3DViewer:
    @abstractmethod
    def setViewPointStyle(self, viewPointStyle: ViewPointStyle): ...

    @abstractmethod
    def add(
        self,
        *solids: Solid,
        representation="wireframe",
        lineWidth=0.25,
        showNormals=False,
        normalLength=0.3,
        colormap="viridis",
        reverseColormap=False,
        colorWithPosition=False,
        opacity=1,
        **kwargs,
    ): ...

    @abstractmethod
    def addDataPoints(
        self,
        dataPoints: np.ndarray,
        colormap="rainbow",
        reverseColormap=False,
        scale=0.15,
        scaleWithValue=True,
        asSpheres=True,
    ):
        """'dataPoints' has to be of shape (n, 4) where the second axis is (value, x, y, z)."""
        ...

    @abstractmethod
    def addImage(
        self,
        image: np.ndarray,
        size: tuple = None,
        minCorner: tuple = (0, 0),
        axis: int = 2,
        position: float = 0,
        colormap: str = "viridis",
    ): ...

    @staticmethod
    @abstractmethod
    def showVolumeSlicer(hist3D: np.ndarray, colormap: str = "viridis", interpolate=False, **kwargs): ...

    @abstractmethod
    def show(self): ...
