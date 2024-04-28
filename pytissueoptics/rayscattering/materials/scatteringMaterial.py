import math

import numpy as np

from pytissueoptics.scene.material import RefractiveMaterial


class ScatteringMaterial(RefractiveMaterial):
    def __init__(self, mu_s=0, mu_a=0, g=0, n=1.0):
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.mu_t = self.mu_a + self.mu_s

        if self.mu_t != 0:
            self._albedo = self.mu_a / self.mu_t
        else:
            self._albedo = 0

        self.g = g
        super().__init__(n)

    def getAlbedo(self):
        return self._albedo

    def getScatteringDistance(self):
        if self.mu_t == 0:
            return math.inf

        rnd = 0
        while rnd == 0:
            rnd = np.random.random()
        return -np.log(rnd) / self.mu_t

    def getScatteringAngles(self):
        phi = np.random.random() * 2 * np.pi
        g = self.g
        if g == 0:
            cost = 2 * np.random.random() - 1
        else:
            temp = (1 - g * g) / (1 - g + 2 * g * np.random.random())
            cost = (1 + g * g - temp * temp) / (2 * g)
        return np.arccos(cost), phi

    def __hash__(self):
        return hash((self.mu_s, self.mu_a, self.g, self.n))
