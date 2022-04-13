from CLPhotons import CLPhotons
from CLSource import CLPencilSource

from pytissueoptics.rayscattering.materials import ScatteringMaterial


worldMaterial = ScatteringMaterial(mu_s=5, mu_a=0.1, g=0.8, index=1.4)
print(worldMaterial.getAlbedo())
source = CLPencilSource(N=100, worldMaterial=worldMaterial)
photons = CLPhotons(source.photons, worldMaterial=worldMaterial)
photons.propagate()

print(photons.HOST_photons)
