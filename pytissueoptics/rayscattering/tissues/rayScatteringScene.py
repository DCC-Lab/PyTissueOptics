from typing import List

from pytissueoptics.scene import MayaviViewer, Scene
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
