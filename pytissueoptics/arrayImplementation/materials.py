import numpy as np
from scalars import *


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

    def getScatteringDistance(self, size):
        if self.mu_t == 0:
            return Material.veryFar

        rnd = np.random.rand(size)
        distances = -np.log(rnd) / self.mu_t
        return Scalars(distances)

    def getScatteringAngles(self, size):
        phi = np.random.rand(size) * 2 * np.pi
        g = self.g
        if g == 0:
            cost = 2 * np.random.rand(size) - 1
        else:
            temp = (1 - g * g) / (1 - g + 2 * g * np.random.rand(size))
            cost = (1 + g * g - temp * temp) / (2 * g)
        return Scalars(np.arccos(cost)), Scalars(phi)

    def __repr__(self):
        return "Material: µs={0} µa={1} g={2} n={3}".format(self.mu_s, self.mu_a, self.g, self.index)

    def __str__(self):
        return "Material: µs={0} µa={1} g={2} n={3}".format(self.mu_s, self.mu_a, self.g, self.index)
