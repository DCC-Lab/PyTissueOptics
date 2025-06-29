from typing import List, Union, Optional

import numpy as np

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Scene, Vector, get3DViewer
from pytissueoptics.scene.detectors import Detector
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer.displayable import Displayable


class ScatteringScene(Scene):
    def __init__(self, solids: List[Solid], detectors: Optional[List[Detector]] = None, worldMaterial=ScatteringMaterial(), ignoreIntersections: bool = False):
        self._detectors = detectors or []
        super().__init__(solids, worldMaterial=worldMaterial, ignoreIntersections=ignoreIntersections)

    def addDetector(self, detector: Detector):
        self._validateLabel(detector.solid)
        detector.solid.setOutsideEnvironment(detector.solid.getEnvironment())

        self._detectors.append(detector)

    @property
    def solids(self):
        return self._solids + [d.solid for d in self._detectors]

    def add(self, solid: Solid, position: Vector = None):
        polygonSample = solid.getPolygons()[0]
        if not isinstance(polygonSample.insideEnvironment.material, ScatteringMaterial):
            raise Exception(
                f"Solid '{solid.getLabel()}' has no ScatteringMaterial defined. "
                f"This is required for any RayScatteringScene. "
            )
        super().add(solid, position)

    def show(self, source: Displayable = None, opacity=0.8, colormap="cool", **kwargs):
        viewer = get3DViewer()
        self.addToViewer(viewer, opacity=opacity, colormap=colormap, **kwargs)
        if source:
            source.addToViewer(viewer)
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
