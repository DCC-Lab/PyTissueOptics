from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Cuboid, Vector, MayaviViewer


class PhantomTissue(RayScatteringScene):
    """ Phantom tissue consisting of 3 layers with various optical properties. """
    TISSUE = []

    def __init__(self):
        self._create()
        super().__init__(self.TISSUE)

    def _create(self):
        n = [1.4, 1.7, 1.4]
        mu_s = [2, 3, 2]
        mu_a = 1
        g = 0.8

        w = 3
        t = [0.75, 0.5, 0.75]

        frontLayer = Cuboid(w, w, t[0], material=ScatteringMaterial(mu_s[0], mu_a, g, n[0]), label="frontLayer")
        middleLayer = Cuboid(w, w, t[1], material=ScatteringMaterial(mu_s[1], mu_a, g, n[1]), label="middleLayer")
        backLayer = Cuboid(w, w, t[2], material=ScatteringMaterial(mu_s[2], mu_a, g, n[2]), label="backLayer")
        layerStack = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        layerStack.translateTo(Vector(0, 0, sum(t) / 2))

        self.TISSUE = [layerStack]

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.TISSUE, representation="surface", colormap="bone", constantColor=False, opacity=0.1)
