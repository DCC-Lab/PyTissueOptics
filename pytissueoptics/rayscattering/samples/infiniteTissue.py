from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene


class InfiniteTissue(ScatteringScene):
    """An infinite tissue with a single material."""

    def __init__(self, material: ScatteringMaterial = ScatteringMaterial(mu_s=2, mu_a=0.1, g=0.8, n=1.4)):
        super().__init__([], worldMaterial=material)
