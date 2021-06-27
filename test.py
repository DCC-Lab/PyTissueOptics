from pytissueoptics import *

world = World()
# We choose a material with scattering properties
mat = Material(mu_s=1, mu_a=0.1, g=0.8, index=1.4)
glass = Material(mu_s=0, mu_a=0.01, g=0, index=1.5)
# mat2 = Material(mu_s=0.1, mu_a=10, g=0.8, index=1.0)
# We want stats: we must determine over what volume we want the energy
stats = Stats(min=(-2, -2, -4), max=(2, 2, 4), size=(50, 50, 50), opaqueBoundaries=False)
# stats2 = Stats(min=(-2, -2, -1), max=(2, 2, 4), size=(50, 50, 50), opaqueBoundaries=False)
# We pick a light source
source = MultimodeFiberSource(direction=zHat, diameter=1, NA=0., index=1, maxCount=10000)
# detector = Detector(NA=0.5)
# We pick a geometry
tissue = Sphere(radius=1, material=glass, stats=stats, label="Sphere")
# tissue = SemiInfiniteLayer(material=mat, stats=stats, label="Layer 1")
# We propagate the photons from the source inside the geometry
world.place(source, position=Vector(0, 0, -4))
world.place(tissue, position=Vector(0, 0, 0))
#world.place(tissue2, position=Vector(0, 0, 2))
#world.place(detector, position=Vector(0, 0, -2))
world.compute(graphs=True)
# Report the results for all geometries
world.report()

