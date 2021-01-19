from vector import *

class Surface:
    def __init__(self, origin, a, b, normal, size = None):
        self.origin = origin
        self.a = a
        self.b = b
        self.normal = normal
        self.size = size
        self.description = "Surface"

    def contains(self, position, epsilon=0.001) -> (bool, float, float):
        local = position-self.origin
        if abs(local.normalizedDotProduct(self.normal)) < epsilon:
            u = local.dot(self.a)
            v = local.dot(self.b) 
            if self.size is None:
                return True, u, v
            else:
                if u > self.size[0] or u < 0 or v > self.size[1] or v < 0:
                    return False, None, None
                else:
                    return True, u, v

        return False, None, None

    def __str__(self):
        return self.description

    def __neg__(self):
        return Surface(self.origin, self.b, self.a, -self.normal, self.size)

class XYPlane(Surface):
    def __init__(self, atZ):
        super(XYPlane, self).__init__(Vector(0,0,atZ), xHat, yHat, zHat)
        self.description = "XY at z={0:.1f}".format(atZ)

class YZPlane(Surface):
    def __init__(self, atX):
        super(YZPlane, self).__init__(Vector(atX,0,0), yHat, zHat, xHat)
        self.description = "YZ at z={0:.1f}".format(atX)

class ZXPlane(Surface):
    def __init__(self, atY):
        super(ZXPlane, self).__init__(Vector(0, atY,0), zHat, xHat, yHat)
        self.description = "ZX at z={0:.1f}".format(atY)

class XYRect(Surface):
    def __init__(self, origin, size):
        super(XYRect, self).__init__(origin, xHat, yHat, zHat, size)
        self.description = "XY at z={0:.1f}".format(origin)

class YZRect(Surface):
    def __init__(self, origin, size):
        super(YZRect, self).__init__(origin, yHat, zHat, xHat, size)
        self.description = "YZ at z={0:.1f}".format(origin)

class ZXRect(Surface):
    def __init__(self, origin, size):
        super(ZXRect, self).__init__(origin, zHat, xHat, yHat, size)
        self.description = "ZX at z={0:.1f}".format(origin)
