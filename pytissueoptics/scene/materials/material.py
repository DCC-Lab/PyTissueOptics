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

    @property
    def isVacuum(self):
        return self.mu_t == 0

    def getScatteringDistance(self):
        if self.mu_t == 0:
            return self.veryFar

        rnd = 0
        while rnd == 0:
            rnd = np.random.random()
        return -np.log(rnd) / self.mu_t
