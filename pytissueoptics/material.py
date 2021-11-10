from pytissueoptics import *
import numpy as np


class Material:
    veryFar = 1e4

    def __init__(self, mu_s=0, mu_a=0, g=0, index=1.0):
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.mu_t = self.mu_a + self.mu_s

        if self.mu_t != 0:
            self.albedo = self.mu_a / self.mu_t
        else:
            self.albedo = 0

        self.g = g
        self.index = index

    def getScatteringDistance(self, photon):
        if self.mu_t == 0:
            return Material.veryFar

        rnd = 0
        while rnd == 0:
            rnd = np.random.random()
        return -np.log(rnd) / self.mu_t

    def getManyScatteringDistances(self, photons):
        if photons.isRowOptimized():
            return Scalars([self.getScatteringDistance(p) for p in photons])

        elif photons.isColumnOptimized():
            rnd = False
            d = Scalars(np.random.random(len(pĥotons)))
            while rnd is False:
                d.conditional_eq(0, np.random.random(), d.v)
                rnd = d.all()
            return Scalars(-np.log(d.v) / self.mu_t)

    def getScatteringAngles(self, photon):
        phi = np.random.random() * 2 * np.pi
        g = self.g
        if g == 0:
            cost = 2 * np.random.random() - 1
        else:
            temp = (1 - g * g) / (1 - g + 2 * g * np.random.random())
            cost = (1 + g * g - temp * temp) / (2 * g)
        return np.arccos(cost), phi

    def getManyScatteringAngles(self, photons):
        if photons.isRowOptimized:
            thetas = []
            phis = []

            for photon in photons:
                theta, phi = self.getScatteringAngles(photon)
                thetas.append(theta)
                phis.append(phi)
            return Scalars(thetas), Scalars(phis)

        elif photons.isColumnOptimized:
            phi = np.random.random(N) * 2 * np.pi
            g = self.g
            if g == 0:
                cost = 2 * np.random.random(N) - 1
            else:
                temp = (1 - g * g) / (1 - g + 2 * g * np.random.random(N))
                cost = (1 + g * g - temp * temp) / (2 * g)
            return Scalars(np.arccos(cost)), Scalars(phi)


    def __repr__(self):
        return "Material: µs={0} µa={1} g={2} n={3}".format(self.mu_s, self.mu_a, self.g, self.index)

    def __str__(self):
        return "Material: µs={0} µa={1} g={2} n={3}".format(self.mu_s, self.mu_a, self.g, self.index)
