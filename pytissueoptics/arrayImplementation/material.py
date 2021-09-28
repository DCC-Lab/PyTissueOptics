import numpy as np
from scalars import Scalars

def isIterable(someObject):
    try:
        iter(someObject)
    except TypeError as te:
        return False
    return True


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

    def getScatteringDistances(self, N):
        rnd = False
        d = Scalars(np.random.random(N))
        while rnd is False:
            d.conditional_eq(0, np.random.random(), d.v)
            rnd = d.all()
        return Scalars(-np.log(d.v) / self.mu_t)

    def getScatteringAngles(self, N):
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
