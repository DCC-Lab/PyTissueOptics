import matplotlib.pyplot as plt
from arrayImplementation.photons import Photons
from arrayImplementation.material import Material
from arrayImplementation.stats import Stats
from vectors import Vectors
from time import time_ns



time0 = time_ns()
batches = 10
N = 5000
mat = Material(mu_s=300, mu_a=5, index=1.3)
stats = Stats()
isAlive = True

for i in range(batches):
    time1 = time_ns()
    photons = Photons(positions=Vectors([[0,0,0]]*N), directions=Vectors([[0,0,1]]*N))
    if i == 0:
        time2 = time_ns()
        print(f"Photons Initialization Approx  ::  {(time2 - time1) / 1000000000}s")

    while isAlive:
        d = mat.getScatteringDistances(photons.N)
        photons.moveBy(d)
        deltas = photons.weight * mat.albedo
        photons.decreaseWeightBy(deltas)
        theta, phi = mat.getScatteringAngles(photons.N)
        photons.scatterBy(theta, phi)
        photons.roulette()
        isAlive = photons.weight.any()

    isAlive = True
    time3 = time_ns()
    stats.showEnergy2D(plane='xz', integratedAlong='y')
    print(f"Batch #{i}   ::  {(time3-time1)/1000000000}s ::  {(i+1)*N/1000000}M photons/{N*batches/1000000}M    ::  ({int((i+1)*N*100/(N*batches))}%)")


time4 = time_ns()
print(f"Complete Process    ::  {N*batches/1000000}M photons    ::  {(time4-time0)/1000000000}s")
print(f"Performances    ::  {((time4-time0)/1000)/(N*batches)} us/photon")
plt.show()
