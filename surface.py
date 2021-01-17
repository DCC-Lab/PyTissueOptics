from vector import *

class Surface:
    def __init__(self, origin, bins = (11,11)):
        self.origin = origin
        self.bins = bins

    def locateIntersection(self, a, b) -> (u,v):
        raise NotImplementedError()

class FlatSurface(Surface):
    def __init__(self, origin, u, v, bins = (11,11)):
        super(FlatSurface, self).__init__(origin, bins)
        self.u = u
        self.v = v

    def locateIntersection(self, a, b) -> (u,v):
        raise NotImplementedError()

class SphericalSurface(Surface):
    def __init__(self, origin, radius, bins = (11,11)):
        super(SphericalSurface, self).__init__(origin, bins)
        self.radius = radius

    def locateIntersection(self, a, b) -> (u,v):
        raise NotImplementedError()
