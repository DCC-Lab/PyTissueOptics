from pytissueoptics import *

# choose the implementation (doesn't work, has to be changed from the file)
# Photons = NativePhotons
# Vectors = NativeVectors
# Scalars = NativeScalars

# Physics
mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.0)
stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
source = IsotropicSource(maxCount=10000)
tissue = Layer(thickness=1, material=mat, stats=stats)
tissue.propagateMany(source)
stats.showEnergy2D(plane='xz', integratedAlong='y', title="NumpyPhotons XZ w/ 10k isotropicSource")

