from vectors import Vectors
from scalars import Scalars


class Photons:
    def __init__(self, positions=Vectors(N=1000), directions=Vectors([zHat]*1000)):
        self.N = len(positions)
        self.r = positions
        self.ez = directions
        self.er = self.ez.anyPerpendicular()
        self.wavelength = None
        self.weight = Scalars([1]*self.N)
        self.path = None
        self.origin = Vectors(N=self.N)
        self._iteration = 0

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return Photon(position=self.r[index], direction=self.ez[index], weight=self.weight[index])

    def __setitem__(self, index, photon):
        self.r[index] = photon.r
        self.z[index] = photon.z
        self.ez[index] = photon.ez
        self.er[index] = photon.er
        self.weight[index] = photon.weight
        self.origin[index] = photon.origin

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < self.N:
            result = self[self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    @property
    def localPosition(self):
        return self.r

    @property
    def globalPosition(self):
        return self.r + self.origin

    @property
    def el(self):
        return self.ez.cross(self.er)

    @property
    def areAllDead(self):
        return all(self.weight == 0)

    @property
    def isAlive(self):
        return self.weight > 0

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin

    def moveBy(self, d):
        self.r.addScaled(self.ez, d)

    def scatterBy(self, theta, phi):
        self.er.rotateAround(self.ez, phi)
        self.ez.rotateAround(self.er, theta)

    def decreaseWeightBy(self, deltas):
        self.weight -= deltas
        self.weight.conditional_lt(0, 0, self.weight.v)

    def roulette(self):
        chance = 0.1
        rouletteMask = self.weight <= 1e-4
        photonsKillMask = (Scalars.random(self.N)) > chance
        photonsKillMask = rouletteMask.logical_and(photonsKillMask)
        self.removePhotonsWeights(photonsKillMask)

    def removePhotonsWeights(self, killMask):
        self.weight = self.weight * ~killMask

    def deflect(self):
        pass

    def reflect(self):
        pass

    def refract(self):
        pass
