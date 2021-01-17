from vector import *
from material import *
from photon import *
from geometry import *

# We choose a material with scattering properties
mat    = Material(mu_s=30, mu_a = 0.5, g = 0.8)

# We determine over what volume we want the statistics
stats  = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))

# We pick a geometry
tissue = Geometry(material=mat, stats=stats)

# We pick a light source
source = IsotropicSource(position=Vector(0,0,0), maxCount=1000)

# We propagate the photons from the source inside the geometry
tissue.propagateMany(source, showProgressEvery=100)

# Report the results
tissue.report()