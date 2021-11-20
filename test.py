from pytissueoptics import *



# Physics
mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.0)
stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
source = Photons(list(IsotropicSource(maxCount=10000)))
tissue = Layer(thickness=2, material=mat, stats=stats)
tissue.origin = Vector(0, 0, -1)
tissue.propagateMany(source)
stats.showEnergy2D(plane='xz', integratedAlong='y', title="NativePhotons XZ w/ 10k isotropicSource")
tissue.report(totalSourcePhotons=len(source))

