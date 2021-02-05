from pytissueoptics import *

# We choose a material with scattering properties
mat    = Material(mu_s=30, mu_a = 0.1, g = 0.8, index = 1.4)
mat2    = Material(mu_s=10, mu_a = 1, g = 0.8, index = 1.4)

# We want stats: we must determine over what volume we want the energy
stats  = Stats(min = (-4, -4, -4), max = (4, 4, 4), size = (20,20,20))
stats2  = Stats(min = (-4, -4, -4), max = (4, 4, 4), size = (20,20,20))

# We pick a light source
source = PencilSource(position=Vector(0,0,-1), direction=Vector(0,0,1), maxCount=1000)

# We pick a geometry
tissue = Layer(position=Vector(0,0,0), thickness=1, material=mat, stats=stats)
tissue2 = Layer(position=Vector(0,0,2), thickness=2, material=mat2, stats=stats2)
tissue2.label = "Layer2"
# tissue3 = Box(position=Vector(0,0,5), size=(1,1,1), material=mat, stats=stats)
#tissue = Sphere(radius=2, material=mat, stats=stats)

# We propagate the photons from the source inside the geometry
World.propagateAll(graphs=True)

# Report the results for all geometries
World.report()
