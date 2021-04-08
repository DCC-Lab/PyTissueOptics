from pytissueoptics import *

# We need to create a clean, empty world to work with
world = World()

# We choose a material with scattering properties
mat = Material(mu_s=30, mu_a=0.1, g=0.8, index=1.4)

# We want stats: we must determine over what volume we want the energy
stats = Stats(min=(-2, -2, -2), max=(2, 2, 2), size=(50, 50, 50))

# We pick a light source
source = PencilSource(direction=zHat, maxCount=10000)
detector = Detector(NA=0.5)

# We pick a geometry
tissue = Layer(thickness=1, material=mat, stats=stats)

# We propagate the photons from the source inside the geometry
world.place(source, position=Vector(0, 0, -1))
world.place(tissue, position=Vector(0, 0, 0))
# world.place(detector, position=Vector(0, 0, -2))

world.compute(graphs=True)

# Report the results for all geometries
world.report()
