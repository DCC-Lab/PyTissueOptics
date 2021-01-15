import numpy as np
import matplotlib.pyplot as plt
from stats import *

class Material:
    def __init__(self, mu_s, mu_a, g):
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.mu_t = self.mu_a + self.mu_s
        self.g = g
        self.stats = Stats()

    def getScatteringDistance(self, photon) -> float:
        rnd = 0
        while rnd == 0:
            rnd = np.random.random()
        return -np.log(rnd)/self.mu_t

    def getScatteringAngles(self, photon) -> (float, float):
        phi = np.random.random()*2*np.pi
        g = self.g
        if g == 0:
            cost = 2*np.random.random()-1 
        else:
            temp = (1-g*g)/(1-g+2*g*np.random.random())
            cost = (1+g*g - temp*temp)/(2*g)
        return (np.arccos(cost), phi)

    def absorbEnergy(self, photon):
        delta = photon.weight * self.mu_a/self.mu_t
        photon.decreaseWeightBy(delta)
        if self.stats is not None:
            self.stats.score(photon, delta)


