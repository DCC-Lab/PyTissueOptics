from pytissueoptics.rayscattering.opencl.CLBasicSimulation import CLBasicSimulation
from pytissueoptics.rayscattering.opencl.CLSource import CLPencilSource
from pytissueoptics.rayscattering.opencl.CLStatistics import CLStatistics
import matplotlib.pyplot as plt
import numpy as np
from pytissueoptics.rayscattering.materials import ScatteringMaterial


worldMaterial = ScatteringMaterial(mu_s=30, mu_a=0.8, g=0.8, index=1.4)
stats = CLStatistics()

source = CLPencilSource(N=500000)
mcSimulation = CLBasicSimulation(source, worldMaterial, stats=stats, stepSize=100)
mcSimulation.launch()

energySum = stats.energy.sum(axis=1)
plt.imshow(np.log(energySum + 0.0001), cmap='viridis',
                          extent=[stats.minBounds[2], stats.maxBounds[2], stats.minBounds[0], stats.maxBounds[0]],
                          aspect='auto')
plt.pause(20)

# stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons")
# print(mcSimulation.HOST_photons)
