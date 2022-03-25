from pytissueoptics.rayscattering.tissues.tissue import Tissue
from pytissueoptics.scene import Cuboid, Material, Vector, MayaviViewer


class PhantomTissue(Tissue):
    """ Phantom tissue consisting of 3 layers with various optical properties. """
    TISSUE = []

    def __init__(self):
        self._create()
        super().__init__(self.TISSUE)

    def _create(self):
        mu_s = [2, 3, 4]
        mu_a = 2
        g = 0.8

        topLayer = Cuboid(4, 4, 0.75, material=Material(mu_s[0], mu_a, g, 1.4))
        middleLayer = Cuboid(4, 4, 0.5, material=Material(mu_s[1], mu_a, g, 1.8))
        bottomLayer = Cuboid(4, 4, 0.75, material=Material(mu_s[2], mu_a, g, 2.0))
        layerStack = bottomLayer.stack(middleLayer, 'Front').stack(topLayer, 'Front')
        layerStack.translateTo(Vector(0, 0, 1))

        self.TISSUE = [layerStack]

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.TISSUE, representation="surface", colormap="bone", constantColor=False, opacity=0.2)
