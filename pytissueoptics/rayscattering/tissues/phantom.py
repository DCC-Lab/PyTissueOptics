from typing import List

from pytissueoptics.rayscattering.tissues.tissue import Tissue
from pytissueoptics.scene import Cuboid, Material, Vector, MayaviViewer
from pytissueoptics.scene.solids import Solid


class PhantomTissue(Tissue):
    """ Phantom tissue consisting of 3 layers with increasing scattering and constant absorption. """
    TISSUE = []

    def __init__(self):
        self._create()
        self._solids = self.TISSUE

    @property
    def solids(self) -> List[Solid]:
        return self._solids

    def _create(self):
        mu_s = [1, 2, 3]
        mu_a = 2
        g = 0.8

        topLayer = Cuboid(4, 4, 0.75, material=Material(mu_s[0], mu_a, g, 1.5))
        middleLayer = Cuboid(4, 4, 0.5, material=Material(mu_s[1], mu_a, g, 2))
        bottomLayer = Cuboid(4, 4, 0.75, material=Material(mu_s[2], mu_a, g, 2.5))
        layerStack = bottomLayer.stack(middleLayer, 'Front').stack(topLayer, 'Front')
        layerStack.translateTo(Vector(0, 0, 1))

        # fixme: intersection finder doesnt support cuboids that were stacked, you need to pass in the cuboidStack.
        #  because stacked cuboids will lose reference to their initial stack surface (not a closed solid anymore).
        #  maybe leave the deleted surface in stacked cuboids or raise when trying to use a "deprecated" cuboid.
        self.TISSUE = [layerStack]

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.TISSUE, representation="surface", colormap="bone", constantColor=False, opacity=0.2)
