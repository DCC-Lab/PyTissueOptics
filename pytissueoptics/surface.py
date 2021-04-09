from .vector import *


class Surface:
    def __init__(self, origin, a, b, normal, size=None, description=None):
        self.origin = ConstVector(origin)
        self.a = ConstVector(a)
        self.b = ConstVector(b)
        self.normal = ConstUnitVector(normal)
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
        local = position - self.origin
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
                if u > self.size[0] + epsilon or u < -epsilon or v > self.size[1] + epsilon or v < -epsilon:
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

        d = (self.origin - position).dot(self.normal) / dotProduct
        if 0 <= d <= maxDistance:
            inPlane = (position - self.origin) + d * direction
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
        s = Surface(Vector(self.origin), Vector(self.b), Vector(self.a), -Vector(self.normal), self.size)
        s.description = self.description
        return s


class XYPlane(Surface):
    def __init__(self, atZ, description=None):
        if description is None:
            description = "XY at z={0:.1f}".format(atZ)
        super(XYPlane, self).__init__(Vector(0, 0, atZ), xHat, yHat, zHat,
                                      size=None, description=description)


class YZPlane(Surface):
    def __init__(self, atX, description=None):
        if description is None:
            description = "YZ at x={0:.1f}".format(atX)
        super(YZPlane, self).__init__(Vector(atX, 0, 0), yHat, zHat, xHat, size=None, description=description)


class ZXPlane(Surface):
    def __init__(self, atY, description=None):
        if description is None:
            description = "ZX at y={0:.1f}".format(atY)
        super(ZXPlane, self).__init__(Vector(0, atY, 0), zHat, xHat, yHat, size=None, description=description)


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

class FresnelIntersect:
    def __init__(self, direction, surface, distance, geometry=None):
        self.surface = surface
        self.direction = direction
        self.distance = distance
        self.geometry = geometry

        assert(self.surface.indexOutside != self.surface.indexInside)

        if direction.dot(surface.normal) < 0:
            # We are going towards inside of the object
            self.actualNormal = -self.surface.normal
            self.indexIn = self.surface.indexOutside
            self.indexOut = self.surface.indexInside
        else:
            # We are going towards outside of the object
            self.actualNormal = self.surface.normal
            self.indexIn = self.surface.indexInside
            self.indexOut = self.surface.indexOutside

        plane = direction.cross(self.actualNormal)
        if plane.normSquared() < 1e-7:
            # Normal incidence: any plane will
            self.incidencePlane = self.direction.anyUnitaryPerpendicular()
        else:
            self.incidencePlane = plane.normalize()

        self.thetaIn = direction.angleWith(self.actualNormal, self.incidencePlane)
        self.thetaOut = None

    def reflectionCoefficient(self, theta) -> float:
        """ Fresnel reflection coefficient, directly from MCML code in 
        Wang, L-H, S.L. Jacques, L-Q Zheng: 
        MCML - Monte Carlo modeling of photon transport in multi-layered
        tissues. Computer Methods and Programs in Biomedicine 47:131-146, 1995. 

        """
        n1 = self.indexIn
        n2 = self.indexOut

        if n1 == n2:
            return 0

        if theta == 0:
            R = (n2-n1)/(n2+n1)
            return R*R

        sa1 = math.sin(theta)
        if sa1*n1/n2 > 1:
            return 1

        sa2 = sa1*n1/n2
        ca1 = math.sqrt(1-sa1*sa1)
        if 1-sa2*sa2 > 0:
            ca2 = math.sqrt(1-sa2*sa2)
        else:
            ca2 = 0
        cap = ca1*ca2 - sa1*sa2 # c+ = cc - ss.
        cam = ca1*ca2 + sa1*sa2 # c- = cc + ss. 
        sap = sa1*ca2 + ca1*sa2 # s+ = sc + cs. 
        sam = sa1*ca2 - ca1*sa2 # s- = sc - cs. 
        r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam); 
        return r

    def isReflected(self) -> bool:
        R = self.reflectionCoefficient(self.thetaIn)
        if np.random.random() < R:
            return True
        return False

    @property
    def reflectionDeflection(self) -> float:
        return 2*self.thetaIn-np.pi

    @property
    def refractionDeflection(self) -> float:
        """ Refract the photon when going through surface.  The surface
        normal in the class Surface always points outward for the object.
        Hence, to simplify the math, we always flip the normal to have 
        angles between -90 and 90.

        Since having n1 == n2 is not that rare, if that is the case we 
        know there is no refraction, and we simply return.
        """

        sinThetaOut = self.indexIn*math.sin(self.thetaIn)/self.indexOut
        if abs(sinThetaOut) > 1:
            # We should not be here.
            raise ValueError("Can't refract beyond angle of total reflection")

        self.thetaOut = np.arcsin(sinThetaOut)
        return self.thetaIn - self.thetaOut
