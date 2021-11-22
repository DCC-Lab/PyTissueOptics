from pytissueoptics import *
from numpy import linspace
import matplotlib.pyplot as plt
N = 100
M = 20
musValues = [10**i for i in linspace(0,2,M)]
muaValues = [10**i for i in linspace(-2,0,M)]

# # Physics
# mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.0)
# stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
# source = Photons(list(IsotropicSource(maxCount=10000)))
# tissue = Layer(thickness=2, material=mat, stats=stats)
# tissue.origin = Vector(0, 0, -1)
# tissue.propagateMany(source)
# stats.showEnergy2D(plane='xz', integratedAlong='y', title="NativePhotons XZ w/ 10k isotropicSource")
# tissue.report(totalSourcePhotons=len(source))

results = []

for mu_s in musValues:
    for mu_a in muaValues:
        world = World()
        # We choose a material with scattering properties
        mat = Material(mu_s=mu_s, mu_a=mu_a, g=0.8, index=1.4)
        # We want stats: we must determine over what volume we want the energy
        stats = Stats(min=(-2, -2, -1), max=(2, 2, 4), size=(50, 50, 50), opaqueBoundaries=False)
        # We pick a light source
        source = PencilSource(direction=zHat, maxCount=N)
        # We pick a geometry
        tissue = Layer(thickness=1, material=mat, stats=stats, label="Layer 1")
        world.place(source, position=Vector(0, 0, -1))
        world.place(tissue, position=Vector(0, 0, 0))
        world.compute(graphs=False, progress=False)

        results.append((mu_s, mu_a, stats.absorbance(), stats.energyRMSVolume(), stats.absorbance()/stats.energyRMSVolume()))
        print("{0:03.2f}\t{1:03.2f}\t{2:03.2f}\t{3:03.2f}\t{4:03.2f}".format(mu_s, mu_a, stats.absorbance(), stats.energyRMSVolume(), stats.absorbance()/stats.energyRMSVolume()))

# mus, mua, E, vol = zip(*results)
# plt.plot(mua, vol, 'ko')
# plt.xlabel("Absorption $\mu_a$ [cm$^{-1}$]")
# plt.ylabel("Volume [cm$^3$]")
# # plt.plot(mua, E, 'ko')
# plt.show()