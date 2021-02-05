from pytissueoptics import *

# We choose a material with scattering properties
mat    = Material(mu_s=30, mu_a = 0.01, g = 0.8, index = 1.4)

# We want stats: we must determine over what volume we want the energy
stats  = Stats(min = (-4, -4, -2), max = (4, 4, 2), size = (50,50,50))

# We pick a light source
source = IsotropicSource(maxCount=10000)
World.place(source, position=Vector(0,0,-0.5))

# We pick a geometry
tissue = Layer(thickness=1, material=mat, stats=stats)
#box    = Box(size=(1,1,1), material=mat, stats=stats)

World.place(tissue, position=Vector(0,0,0))

# We propagate the photons from the source inside the geometry
World.compute(graphs=True)

# Report the results for all geometries
World.report()
