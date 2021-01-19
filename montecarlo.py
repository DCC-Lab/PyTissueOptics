from vector import *
from material import *
from photon import *
from geometry import *

# We choose a material with scattering properties
mat    = Material(mu_s=10, mu_a = 0.01, g = 0.8)

# We want stats: we must determine over what volume we want the energy
stats  = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))

# We pick a geometry
#tissue = Box(size=(2,2,2), material=mat, stats=stats)
tissue = Layer(thickness=2, material=mat, stats=stats)
#tissue = Sphere(radius=2, material=mat, stats=stats)

# We pick a light source
source = IsotropicSource(position=Vector(0,0,0.01), maxCount=10000)

# We propagate the photons from the source inside the geometry
# The source needs to be inside the geometry (for now)
tissue.propagateMany(source, showProgressEvery=100)

# Soon, will be like this:
#world.place(tissue, position)
#world.place(source, position)
#world.propagateMany(showProgressEvery=100)

# Report the results
tissue.report()
