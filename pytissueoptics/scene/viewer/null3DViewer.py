import warnings

import numpy as np
from scene.solids import Solid
from scene.viewer.abstract3DViewer import Abstract3DViewer

from pytissueoptics import ViewPointStyle


class Null3DViewer(Abstract3DViewer):
    def setViewPointStyle(self, viewPointStyle: ViewPointStyle):
        pass

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
    ):
        pass

    def addDataPoints(
        self,
        dataPoints: np.ndarray,
        colormap="rainbow",
        reverseColormap=False,
        scale=0.15,
        scaleWithValue=True,
        asSpheres=True,
    ):
        pass

    def addImage(
        self,
        image: np.ndarray,
        size: tuple = None,
        minCorner: tuple = (0, 0),
        axis: int = 2,
        position: float = 0,
        colormap: str = "viridis",
    ):
        pass

    @staticmethod
    def showVolumeSlicer(hist3D: np.ndarray, colormap: str = "viridis", interpolate=False, **kwargs):
        warnings.warn("Attempting to show a volume slicer with a Null3DViewer. No action will be taken.")

    def show(self):
        warnings.warn("Attempting to show a Null3DViewer. No action will be taken.")
