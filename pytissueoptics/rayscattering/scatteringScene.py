from typing import List

import numpy as np

from pytissueoptics import Vector
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import MayaviViewer, Scene
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer.displayable import Displayable


class ScatteringScene(Scene):
    def __init__(self, solids: List[Solid], worldMaterial=ScatteringMaterial(), ignoreIntersections: bool = False):
        super().__init__(solids, worldMaterial=worldMaterial, ignoreIntersections=ignoreIntersections)

    def add(self, solid: Solid, position: Vector = None):
        polygonSample = solid.getPolygons()[0]
        if not isinstance(polygonSample.insideEnvironment.material, ScatteringMaterial):
            raise Exception(f"Solid '{solid.getLabel()}' has no ScatteringMaterial defined. "
                            f"This is required for any RayScatteringScene. ")
        super().add(solid, position)

    def show(self, source: Displayable = None, sourceSize: float = 0.1, **kwargs):
        viewer = MayaviViewer()
        self.addToViewer(viewer, **kwargs)
        if source:
            source.addToViewer(viewer, size=sourceSize)
        viewer.show()

    def getEstimatedIPP(self, weightThreshold: float) -> float:
        """
        Get the estimated number of interactions per photon. This gross estimation is done by assuming an infinite
        medium of mean scene albedo. Used as a starting point for the OpenCL kernel optimization.
        """
        materials = self.getMaterials()
        averageAlbedo = sum([mat.getAlbedo() for mat in materials]) / len(materials)
        estimatedIPP = -np.log(weightThreshold) / averageAlbedo
        return estimatedIPP
