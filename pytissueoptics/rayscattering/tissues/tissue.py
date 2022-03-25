from typing import List

from pytissueoptics.scene import MayaviViewer, Material
from pytissueoptics.scene.solids import Solid


class Tissue:
    def __init__(self, solids: List[Solid]):
        self._solids = solids

    def getSolids(self) -> List[Solid]:
        return self._solids

    def addToViewer(self, viewer: MayaviViewer):
        raise NotImplementedError

    def display(self):
        viewer = MayaviViewer()
        self.addToViewer(viewer)
        viewer.show()

    def setWorldMaterial(self, material: Material):
        for solid in self._solids:
            solid.setOutsideMaterial(material)
