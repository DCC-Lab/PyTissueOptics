from vector import *

class Surface:
    def __init__(self, origin, normal, bins = (11,11)):
        self.origin = origin
        self.normal = normal
        self.bins = bins
        self.intensity = np.zeros(bins)

    def uvCoordinates(self, point) -> (float,float):
        raise NotImplementedError()

    def uvIndexes(self, point) -> (int, int):
        raise NotImplementedError()

    def score(self, photon):
        i,j = self.uvIndexes(photon.r)
        self.scoreAt(photon, i, j)

    def scoreAt(self, photon, i, j):
        self.intensity[i,j] += photon.weight

class FlatSurface(Surface):
    def __init__(self, origin, a, b, bins = (11,11)):
        super(FlatSurface, self).__init__(origin, u.cross(v), bins)
        self.a = a
        self.b = b

    def locateProjection(self, point) -> (float,float):
        raise NotImplementedError()

class SphericalSurface(Surface):
    def __init__(self, origin, radius, bins = (37,11)):
        super(SphericalSurface, self).__init__(origin, None, bins)
        self.radius = radius

