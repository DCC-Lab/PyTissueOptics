from vector import *

class Surface:
    def __init__(self, origin, a, b, normal):
        self.origin = origin
        self.normal = normal
        self.a = a
        self.b = b
        self.description = "Surface"

    def contains(self, position, epsilon=0.001) -> (bool, float, float):
        local = position-self.origin
        if abs(local.normalizedDotProduct(self.normal)) < epsilon:
            return True, local.dot(self.a), local.dot(self.b) 
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
    def __init__(self, origin, size, normal):
        super(XYRect, self).__init__(origin, size[0]*xHat, size[1]*yHat, normal)
        self.description = "XY at z={0:.1f}".format(origin)

class YZRect(Surface):
    def __init__(self, origin, size, normal):
        super(YZRect, self).__init__(origin, size[0]*yHat, size[1]*zHat, normal)
        self.description = "YZ at z={0:.1f}".format(origin)

class ZXRect(Surface):
    def __init__(self, origin, size, normal):
        super(ZXRect, self).__init__(origin, size[0]*zHat, size[1]*xHat, normal)
        self.description = "ZX at z={0:.1f}".format(origin)
