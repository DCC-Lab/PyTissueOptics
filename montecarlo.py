from pytissueoptics import *

# We choose a material with scattering properties
mat    = Material(mu_s=30, mu_a = 0.1, g = 0.8, index = 1.4)

# We want stats: we must determine over what volume we want the energy
stats  = Stats(min = (-2, -2, 0), max = (2, 2, 2), size = (20,20,20))

# We pick a light source
source = PencilSource(position=Vector(0,0,-4), direction=Vector(0,0,1), maxCount=10000)

# We pick a geometry
#tissue = Layer(thickness=2, material=mat, stats=stats)
tissue = Box(position=Vector(0,0,0), size=(2,2,2), material=mat, stats=stats)
#tissue = Sphere(radius=2, material=mat, stats=stats)


# We propagate the photons from the source inside the geometry
tissue.propagateAll(graphs=True)

# Report the results for tissue
tissue.report()
