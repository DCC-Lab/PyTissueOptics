from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Cube, MayaviViewer


class CubeTissue(RayScatteringScene):
    """ A big cube with one material."""
    TISSUE = []

    def __init__(self, side: float = 2, material: ScatteringMaterial = ScatteringMaterial(mu_s=2, mu_a=0.1, g=0.8, index=1.4)):
        self.material = material
        self.side = side
        self._create()
        super().__init__(self.TISSUE)

    def _create(self):
        tissue = Cube(self.side, material=self.material, label="cube")
        self.TISSUE = [tissue]

    def addToViewer(self, viewer: MayaviViewer):
        viewer.add(*self.TISSUE, representation="surface", colormap="bone", constantColor=False, opacity=0.1)
