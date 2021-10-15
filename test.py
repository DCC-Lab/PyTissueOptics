from pytissueoptics import *
from numpy import linspace
import matplotlib.pyplot as plt
N = 100
M = 20
musValues = [10**i for i in linspace(0,2,M)]
muaValues = [10**i for i in linspace(-2,0,M)]

# stats = Stats(min=(-2, -2, -1), max=(2, 2, 4), size=(10, 10, 10), opaqueBoundaries=False)
# sum = stats.energy.sum(axis=(0, 2))
# plt.plot(stats.yCoords, np.log10(sum + 0.0001), 'ko--')
# plt.show()
# exit()
results = []

for mu_s in musValues:
    for mu_a in muaValues:
        world = World()
        # We choose a material with scattering properties
        mat = Material(mu_s=mu_s, mu_a=mu_a, g=0.8, index=1.4)
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

        vol = stats.energyVolume()
        results.append((mu_s, mu_a, stats.absorbance(), vol))
        print("{0:03.2f}\t{1:03.2f}\t{2:03.2f}\t{3:03.2f}".format(mu_s, mu_a, stats.absorbance(), vol))

mus, mua, E, vol = zip(*results)
plt.plot(mua, vol, 'ko')
plt.xlabel("Absorption $\mu_a$ [cm$^{-1}$]")
plt.ylabel("Volume [cm$^3$]")
# plt.plot(mua, E, 'ko')
plt.show()