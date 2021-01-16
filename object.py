import numpy as np
import matplotlib.pyplot as plt
from stats import *
from material import *
from vector import *
from photon import *

class Object:
    def __init__(self, material, stats=None):
        self.material = material
        self.stats = stats

    def propagate(self, photon):
        while photon.isAlive and self.contains(photon.r):
            d = self.material.getScatteringDistance(photon)
            (theta, phi) = self.material.getScatteringAngles(photon)
            photon.scatterBy(theta, phi)
            photon.moveBy(d)
            self.absorbEnergy(photon)
            photon.roulette()

    def propagateManyPhotons(self, N, showProgressEvery=100):
        for i in range(1,N+1):
            self.propagate(Photon())
            if i  % showProgressEvery == 0:
                print("Photon {0}/{1}".format(i,N) )
                if self.stats is not None:
                    self.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    def contains(self, position) -> bool:
        return True

    def absorbEnergy(self, photon):
        delta = photon.weight * self.material.albedo
        if self.stats is not None:
            self.stats.score(photon, delta)
        photon.decreaseWeightBy(delta)    

    def report(self):
        if self.stats is not None:
            self.stats.report()
            self.stats.show2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)

