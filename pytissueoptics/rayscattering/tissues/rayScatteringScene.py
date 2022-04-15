from typing import List

from pytissueoptics.scene import MayaviViewer, Material, Scene
from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.solids import Solid


class RayScatteringScene(Scene):
    def __init__(self, solids: List[Solid]):
        super().__init__(solids)

    def addToViewer(self, viewer: MayaviViewer):
        raise NotImplementedError

    def display(self):
        viewer = MayaviViewer()
        self.addToViewer(viewer)
        viewer.show()

    def setWorldMaterial(self, material: Material):
        worldEnvironment = Environment(material)
        for solid in self._solids:
            solid.setOutsideEnvironment(worldEnvironment)
