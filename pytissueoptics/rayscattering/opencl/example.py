import time

from CLPhotons import CLPhotons
from CLSource import CLPencilSource
from stats import Stats
from pytissueoptics.rayscattering.materials import ScatteringMaterial


worldMaterial = ScatteringMaterial(mu_s=30, mu_a=0.1, g=0.8, index=1.4)
print(" === CREATION ===")
t0 = time.time_ns()
source = CLPencilSource(N=10, worldMaterial=worldMaterial)
t1 = time.time_ns()
photons = CLPhotons(source.photons, worldMaterial=worldMaterial)
t2 = time.time_ns()
print("Create CPU Photons: ", (t1-t0)/1e9, "s")
print("Create GPU Photons: ", (t2-t1)/1e9, "s")


photons.propagate()

print(" === RESULTS ===")
t3 = time.time_ns()
for i in photons.logger:
    print(i)
# stats = Stats(photons.logger)
t4 = time.time_ns()
print("Calculate Stats: ", (t4-t3)/1e9, "s")
# stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
