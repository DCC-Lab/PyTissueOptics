from typing import List

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import MayaviViewer, Scene
from pytissueoptics.scene.solids import Solid


class RayScatteringScene(Scene):
    def __init__(self, solids: List[Solid], worldMaterial=ScatteringMaterial()):
        super().__init__(solids, worldMaterial=worldMaterial)

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.solids, representation="surface", colormap="bone", constantColor=False, opacity=0.1)

    def display(self):
        viewer = MayaviViewer()
        self.addToViewer(viewer)
        viewer.show()
