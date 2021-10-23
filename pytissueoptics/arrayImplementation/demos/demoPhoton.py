from pytissueoptics.source import PencilSource
from pytissueoptics.photon import Photon
from pytissueoptics.vector import Vector
from pytissueoptics.material import Material
from time import time_ns


time0 = time_ns()
N = 500
source = PencilSource(direction=Vector([0, 0, 1]), maxCount=N)
mat = Material(mu_s=300, mu_a=5, index=1.3)
time1 = time_ns()
for i, photon in enumerate(source):
    while photon.isAlive:
        # Pick distance to scattering point
        d = mat.getScatteringDistance(photon)
        photon.moveBy(d)
        delta = photon.weight * mat.albedo
        photon.decreaseWeightBy(delta)
        theta, phi = mat.getScatteringAngles(photon)
        photon.scatterBy(theta, phi)
        photon.roulette()
time2 = time_ns()

print(f"Photons Initialization Approx  ::  {(time1 - time0) / 1000000000}s")
print(f"Complete Process    ::  {N/1000000}M photons    ::  {(time2-time1)/1000000000}s")
print(f"Performances    ::  {((time2-time1)/1000)/N} us/photon")