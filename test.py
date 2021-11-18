from pytissueoptics import *

# choose the implementation
Photons = ArrayPhotons
Vectors = NumpyVectors
Scalars = NumpyScalars

# Physics
mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.2)
stats = Stats(min=(-2, -2, -1), max=(2, 2, 4), size=(50, 50, 50), opaqueBoundaries=False)
source = IsotropicSource(maxCount=3000)
tissue = Layer(thickness=1, material=mat, stats=stats, label="Layer 1")
tissue.propagateMany(source)
stats.showEnergy2D(plane='xz', integratedAlong='y', title="Total photons energy")

