from vector import *

class Surface:
    def __init__(self, origin, a, b, normal, size = None, description=None):
        self.origin = origin
        self.a = a
        self.b = b
        self.normal = normal
        self.size = size
        self.indexInside = 1.0
        self.indexOutside = 1.0

        if description is None:
            self.description = "Surface"
        else:
            self.description = description

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
        s = Surface(self.origin, self.b, self.a, -self.normal, self.size)
        s.description = self.description
        return s

class XYPlane(Surface):
    def __init__(self, atZ, description=None):
        if description is None:
            description = "XY at z={0:.1f}".format(atZ)
        super(XYPlane, self).__init__(Vector(0,0,atZ), xHat, yHat, zHat, 
                                      size=None, description=description)

class YZPlane(Surface):
    def __init__(self, atX, description=None):
        if description is None:
            description = "YZ at x={0:.1f}".format(atX)
        super(YZPlane, self).__init__(Vector(atX,0,0), yHat, zHat, xHat, size=None, description=description)

class ZXPlane(Surface):
    def __init__(self, atY, description=None):
        if description is None:
            description = "ZX at y={0:.1f}".format(atY)
        super(ZXPlane, self).__init__(Vector(0, atY,0), zHat, xHat, yHat, size=None, description=description)

class XYRect(Surface):
    def __init__(self, origin, size, description=None):
        if description is None:
            description = "XY at z={0:.1f}".format(origin)
        super(XYRect, self).__init__(origin, xHat, yHat, zHat, size, description=description)

class YZRect(Surface):
    def __init__(self, origin, size, description=None):
        if description is None:
            description = "YZ at x={0:.1f}".format(origin)
        super(YZRect, self).__init__(origin, yHat, zHat, xHat, size, description=description)

class ZXRect(Surface):
    def __init__(self, origin, size, description=None):
        if description is None:
            description = "ZX at y={0:.1f}".format(origin)
        super(ZXRect, self).__init__(origin, zHat, xHat, yHat, size, description)
