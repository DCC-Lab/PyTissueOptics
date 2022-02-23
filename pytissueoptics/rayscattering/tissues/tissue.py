from typing import List

from pytissueoptics.scene import MayaviViewer
from pytissueoptics.scene.solids import Solid


class Tissue:
    @property
    def solids(self) -> List[Solid]:
        raise NotImplementedError

    def addToViewer(self, viewer: MayaviViewer):
        raise NotImplementedError

    def display(self):
        viewer = MayaviViewer()
        self.addToViewer(viewer)
        viewer.show()
