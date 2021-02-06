from pytissueoptics import *

# We choose a material with scattering properties
mat    = Material(mu_s=30, mu_a = 0.1, g = 0.8, index = 1.0)

# We want stats: we must determine over what volume we want the energy
stats  = Stats(min = (-2,-2,-2), max = (2,2,2), size = (50,50,50))

# We pick a light source
source = PencilSource(direction=UnitVector(0,0,1), maxCount=10000)

# We pick a geometry
tissue = Layer(thickness=1, material=mat, stats=stats)

# We propagate the photons from the source inside the geometry
World.place(source, position=Vector(0,0,-1))
World.place(tissue, position=Vector(0,0,0))

World.compute(graphs=True)

# Report the results for all geometries
World.report()
