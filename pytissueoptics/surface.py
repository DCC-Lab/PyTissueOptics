from .vector import *

class Surface:
    def __init__(self, origin, a, b, normal, size = None, description=None):
        self.origin = origin
        self.a = a
        self.b = b
        self.normal = normal
        self.size = size
        self.indexInside = 1.0
        self.indexOutside = 1.0
        self.crossing = []

        if description is None:
            self.description = "Surface"
        else:
            self.description = description

    def contains(self, position, epsilon=0.001) -> (bool, float, float):
        # Convert to position in surface coordinates
        local = position-self.origin
        if abs(local.normalizedDotProduct(self.normal)) < epsilon:
            # We're in the plane of the surface element
            # Compute u,v coordinates
            u = local.dot(self.a)
            v = local.dot(self.b) 
            if self.size is None:
                # If element is infinite, then we really are on it, return coordinates
                return True, u, v
            else:
                # If element is finite, check we are on surface element with a small margin
                if u > self.size[0]+epsilon or u < -epsilon or v > self.size[1] + epsilon or v < -epsilon:
                    return False, None, None
                else:
                    return True, u, v

        return False, None, None

    def intersection(self, position, direction, maxDistance) -> (bool, float): 
        # This function is not as efficient as the Geometry implementation
        # So we don't really use it, but it is still available
        # https://en.wikipedia.org/wiki/Line–plane_intersection

        dotProduct = direction.dot(self.normal)
        if dotProduct == 0:
            return False, maxDistance

        d = (position - self.origin).dot(self.normal)/dotProduct
        if d >= 0 and d <= maxDistance:
            inPlane = (position-self.origin)+d*direction
            if self.contains(inPlane):
                return True, d
            else:                
                return False, maxDistance
        else:
            return False, maxDistance

    def totalWeightCrossing(self):
        weights = []

        for (r, w) in self.crossing:
            weights.append(w)

        return sum(weights)

    def __repr__(self):
        return "'{0}' normal:{1}".format(self.description, self.normal)

    def __str__(self):
        return "{0}".format(self.description, self.normal)

    def __neg__(self):
        s = Surface( Vector(self.origin), Vector(self.b), Vector(self.a), -Vector(self.normal), self.size)
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
