from CLPhotons import CLPhotons
from CLSource import CLPencilSource
from stats import Stats
from pytissueoptics.rayscattering.materials import ScatteringMaterial


worldMaterial = ScatteringMaterial(mu_s=30, mu_a=0.1, g=0.8, index=1.4)
print(worldMaterial.getAlbedo())
source = CLPencilSource(N=10, worldMaterial=worldMaterial)
photons = CLPhotons(source.photons, worldMaterial=worldMaterial)
photons.propagate()

print(photons.HOST_logger)
print(photons.HOST_photons)
stats = Stats(photons.HOST_logger)
stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
