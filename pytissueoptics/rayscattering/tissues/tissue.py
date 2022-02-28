from typing import List

from pytissueoptics.scene import MayaviViewer, Material
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

    def setWorldMaterial(self, material: Material):
        for solid in self.solids:
            solid.setOutsideMaterial(material)
