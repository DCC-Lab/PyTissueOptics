from .photon import *
import time
from numpy import random, pi, cos, sin, tan

class Source:
    def __init__(self, maxCount):
        self.origin = Vector(0, 0, 0)
        self.maxCount = maxCount
        self.iteration = 0
        self._photons = []

    def __iter__(self):
        self.iteration = 0
        return self

    def __len__(self) -> int:
        return self.maxCount

    def __getitem__(self, item):
        if item < 0:
            # Convert negative index to positive (i.e. -1 == len - 1)
            item += self.maxCount

        if item < 0 or item >= self.maxCount:
            raise IndexError(f"Index {item} out of bound, min = 0, max {self.maxCount}.")

        start = time.monotonic()
        while len(self._photons) <= item:
            self._photons.append(self.newPhoton())
            if time.monotonic() - start > 2:
                warnings.warn(f"Generating missing photon. This can take a few seconds.", UserWarning)

        return self._photons[item]

    def __next__(self) -> Photon:
        if self.iteration >= self.maxCount:
            raise StopIteration
        # This should be able to know if enough photon. If not enough, generate them
        photon = self[self.iteration]
        self.iteration += 1
        return photon

    def newPhoton(self) -> Photon:
        raise NotImplementedError()


class IsotropicSource(Source):
    def __init__(self, maxCount):
        super(IsotropicSource, self).__init__(maxCount)

    def newPhoton(self) -> Photon:
        p = Photon()
        p.r = self.origin

        phi = np.random.random() * 2 * np.pi
        cost = 2 * np.random.random() - 1

        p.scatterBy(np.arccos(cost), phi)
        return p


class PencilSource(Source):
    def __init__(self, direction, maxCount):
        super(PencilSource, self).__init__(maxCount)
        self.direction = Vector(direction)

    def newPhoton(self) -> Photon:
        return Photon(Vector(self.origin), Vector(self.direction))


class MultimodeFiberSource(Source):
    def __init__(self, direction, diameter, NA, index, maxCount):
        super(MultimodeFiberSource, self).__init__(maxCount)
        self.direction = UnitVector(direction)
        self.xAxis = UnitVector(self.direction.anyPerpendicular())
        self.yAxis = UnitVector(self.direction.cross(self.xAxis))
        self.radius = diameter / 2
        self.NA = NA
        self.index = index

    @property
    def maxAngle(self):
        return math.asin(self.NA / self.index)

    def newPhoton(self) -> Photon:
        positionVector = self.newUniformPosition()
        directionVector = self.newUniformConeDirection()

        return Photon(Vector(positionVector), Vector(directionVector))

    def newUniformPosition(self):

        position = None
        while position is None:
            x = self.radius * (2 * random.random() - 1)
            y = self.radius * (2 * random.random() - 1)

            if x * x + y * y < self.radius * self.radius:
                position = Vector.fromScaledSum(self.origin, self.xAxis, x)
                position += self.yAxis * y

        return position

    def newUniformConeDirection(self):
        # Generating a uniformly distributed random vector in a cone is funky:
        # https://math.stackexchange.com/questions/56784/generate-a-random-direction-within-a-cone

        z = random.uniform(math.cos(self.maxAngle), 1)
        theta1 = math.acos(z)
        theta2 = 2 * pi * random.random()
        beta = (pi / 2) - theta1
        a = z / tan(beta)
        x = cos(theta2) * a
        y = sin(theta2) * a

        return UnitVector(x, y, z)
