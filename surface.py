from vector import *

class Surface:
    def __init__(self, origin, normal, bins = (11,11)):
        self.origin = origin
        self.normal = normal
        self.bins = bins
        self.intensity = np.zeros(bins)

    def locateProjection(self, point) -> (float,float):
        raise NotImplementedError()

    def locateIntersection(self, a, b) -> (float,float):
        raise NotImplementedError()

    def score(self, photon):
        u,v = self.locateProjection(photon.r)
        self.scoreAt(photon, u, v)

    def scoreAt(self, photon, u, v):
        self.intensity[u,v] += photon.weight

class FlatSurface(Surface):
    def __init__(self, origin, u, v, bins = (11,11)):
        super(FlatSurface, self).__init__(origin, u.cross(v), bins)
        self.u = u
        self.v = v

    def locateProjection(self, point) -> (float,float):
        raise NotImplementedError()

    def locateIntersection(self, a, b) -> (float,float):
        raise NotImplementedError()

class SphericalSurface(Surface):
    def __init__(self, origin, radius, bins = (37,11)):
        super(SphericalSurface, self).__init__(origin, None, bins)
        self.radius = radius

