import numpy as np
from vector import *
from material import *
from photon import *
from object import *

if __name__ == "__main__":
    mat    = Material(mu_s=30, mu_a = 0.5, g = 0.8)
    stats  = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))
    tissue = Object(material=mat, stats=stats)

    tissue.propagateMany(N=1000, showProgressEvery=100)

    # for i in range(1,N+1):
    #     tissue.propagate(Photon())
    #     if i  % 100 == 0:
    #         print("Photon {0}/{1}".format(i,N) )
    #         if stats is not None:
    #             #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(i))
    #             stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i))
    #             #stats.show1D(axis='z', title="{0} photons".format(i))

    if stats is not None:
        stats.report()
        stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(N), realtime=False)
        #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)
