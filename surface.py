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

class XYPlane(Surface):
    def __init__(self, atZ, normal):
        super(XYPlane, self).__init__(Vector(0,0,atZ), xHat, yHat, normal)
        self.description = "XY at z={0:.1f}".format(atZ)

class YZPlane(Surface):
    def __init__(self, atX, normal):
        super(YZPlane, self).__init__(Vector(atX,0,0), yHat, zHat, normal)
        self.description = "YZ at z={0:.1f}".format(atX)

class ZXPlane(Surface):
    def __init__(self, atY, normal):
        super(ZXPlane, self).__init__(Vector(0, atY,0), zHat, xHat, normal)
        self.description = "ZX at z={0:.1f}".format(atY)

class XYRect(Surface):
    def __init__(self, origin, normal, size):
        super(XYRect, self).__init__(origin, xHat, yHat, normal, size)
        self.description = "XY at z={0:.1f}".format(origin)

class YZRect(Surface):
    def __init__(self, origin, normal, size):
        super(YZRect, self).__init__(origin, yHat, zHat, normal, size)
        self.description = "YZ at z={0:.1f}".format(origin)

class ZXRect(Surface):
    def __init__(self, origin, normal, size):
        super(ZXRect, self).__init__(origin, zHat, xHat, normal, size)
        self.description = "ZX at z={0:.1f}".format(origin)
