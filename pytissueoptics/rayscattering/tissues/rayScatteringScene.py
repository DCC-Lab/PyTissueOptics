from typing import List

from pytissueoptics import Vector
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import MayaviViewer, Scene
from pytissueoptics.scene.solids import Solid


class RayScatteringScene(Scene):
    def __init__(self, solids: List[Solid], worldMaterial=ScatteringMaterial()):
        super().__init__(solids, worldMaterial=worldMaterial)

    def add(self, solid: Solid, position: Vector = None):
        polygonSample = solid.getPolygons()[0]
        if not isinstance(polygonSample.insideEnvironment.material, ScatteringMaterial):
            raise Exception(f"Solid '{solid.getLabel()}' has no ScatteringMaterial defined. "
                            f"This is required for any RayScatteringScene. ")
        super().add(solid, position)

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.solids, representation="surface", colormap="bone", constantColor=False, opacity=0.1)

    def display(self):
        viewer = MayaviViewer()
        self.addToViewer(viewer)
        viewer.show()
