import numpy as np
from vector import *
from material import *
from photon import *

if __name__ == "__main__":
    N = 1000
    mat = Material(mu_s=30, mu_a = 0.5, g = 0.8)
    
    try:
        mat.stats.restore("output.json")
    except:
        mat.stats = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))

    for i in range(1,N+1):
        photon = Photon()
        while photon.isAlive:
            d = mat.getScatteringDistance(photon)
            (theta, phi) = mat.getScatteringAngles(photon)
            photon.scatterBy(theta, phi)
            photon.moveBy(d)
            mat.absorbEnergy(photon)
            photon.roulette()
        if i  % 100 == 0:
            print("Photon {0}/{1}".format(i,N) )
            if mat.stats is not None:
                #mat.stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(i))
                mat.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i))
                #mat.stats.show1D(axis='z', title="{0} photons".format(i))

    if mat.stats is not None:
        mat.stats.report()
        mat.stats.save("output.json")
        mat.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(N), realtime=False)
        mat.stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)
