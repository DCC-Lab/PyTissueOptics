from pytissueoptics import *

N = 100

world = World()
# We choose a material with scattering properties
mat = Material(mu_s=30, mu_a=0.1, g=0.8, index=1.4)
# mat2 = Material(mu_s=0.1, mu_a=10, g=0.8, index=1.0)
# We want stats: we must determine over what volume we want the energy
stats = Stats(min=(-2, -2, -1), max=(2, 2, 4), size=(50, 50, 50), opaqueBoundaries=False)
# stats2 = Stats(min=(-2, -2, -1), max=(2, 2, 4), size=(50, 50, 50), opaqueBoundaries=False)
# We pick a light source
source = PencilSource(direction=zHat, maxCount=N)
detector = Detector(NA=0.5)
# We pick a geometry
tissue = Layer(thickness=1, material=mat, stats=stats, label="Layer 1")
# tissue2 = Layer(thickness=1, material=mat2, stats=stats2, label="Layer 2")
# We propagate the photons from the source inside the geometry
world.place(source, position=Vector(0, 0, -1))
world.place(tissue, position=Vector(0, 0, 0))
#world.place(tissue2, position=Vector(0, 0, 2))
#world.place(detector, position=Vector(0, 0, -2))
world.compute(graphs=False, progress=False)
# Report the results for all geometries
print(stats.transmittance(tissue.surfaces))
# world.report(graphs=False)

