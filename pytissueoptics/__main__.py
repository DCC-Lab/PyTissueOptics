from pytissueoptics import *

# We choose a material with scattering properties
mat    = Material(mu_s=30, mu_a = 0.1, g = 0.8, index = 1.4)

# We want stats: we must determine over what volume we want the energy
stats  = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))

# We pick a geometry
tissue = Layer(thickness=2, material=mat, stats=stats)
#tissue = Box(size=(2,2,2), material=mat, stats=stats)
#tissue = Sphere(radius=2, material=mat, stats=stats)

# We pick a light source
# The source needs to be inside the geometry (for now)
source = PencilSource(position=Vector(0,0,0.001), direction=Vector(0,0,1), maxCount=1000)

# We propagate the photons from the source inside the geometry
tissue.propagateMany(source, graphs=True)

# Report the results
tissue.report()
