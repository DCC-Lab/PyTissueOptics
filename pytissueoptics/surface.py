from .vector import *
from numpy import sqrt, isnan, arctan2, linspace, polyfit

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


class AsphericSurface(Surface):
    """
    An aspheric dielectric interface of radius R, and a conical factor kappa.
    The plan here is simple: in the paraxial approximation, any conical
    surface will look like a spherical surface, therefore we set the elements
    the exactly that: a spherical surface of radius R. For all calculations
    involving aperture stops, image conjuagtes, etc... we will always use the
    paraxial form of this matrix (the non-paraxial rays are just a
    correction). However, when we trace the rays, then that's a different
    story: we will use the mul_ray_nonparaxial function that is not a matrix
    multiplication but close to the axis, it gives similar results to the
    mul_ray_paraxial function.

    We use the following definitions:
    https://en.wikipedia.org/wiki/Aspheric_lens

    Parameters
    ----------
    R : float (Optional, default infinity)
        The radius of the dielectric interface
    kappa : float (default 0, sphere)
        The conical parameter of the interface
        kappa < -1    : hyperbola
        kappa == -1   : parabola
        -1 < kappa < 0: prolate ellipse
        kappa == 0    : sphere
        kappa > 0     : oblate ellipse
    Notes
    -----
    A convex interface from the perspective of the ray has R > 0
    """

    def __init__(self, R, kappa, description=None):
        super(AsphericSurface, self).__init__(origin=oHat, a=xHat, b=yHat, normal=zHat, size=(1,1), description=description)
        self.kappa = kappa
        self.R = R

    def contains(self, position, epsilon=0.001) -> (bool, float, float):
        local = position - self.origin
        x = local.x
        y = local.y
        z = local.z

        zSurface = self.z(x,y)
        if zSurface > z:
            return True
        else:
            return False

    def segmentValidityAboveSurface(self, position, direction, maxDistance):
        d = direction*maxDistance
        distance = d.abs()
        direction = d.normalized()

        (u,v,w) = (d.x, d.y, d.z)
        (xo, yo, zo) = (position.x, position.y, position.z)

        # quadratic equation domain validity
        a = u*u+v*v
        b = (2*xo*u + 2*yo*v)
        c = (xo*xo + yo*yo - self.R*self.R/(1+self.kappa))

        delta = b*b-4*a*c

        # Because of roundoff errors, I need to add a small espilon.
        epsilon = 0.0000001

        validRange = [0,1]

        if delta < 0:
            tMinus = None
            tPlus = None
            validRange = None
        else:
            # tMinus is always smaller than tPlus
            tMinus = (-b-sqrt(delta))/2/a
            tPlus = (-b+sqrt(delta))/2/a
            
            if tMinus > 1 or tPlus < 0:
                validRange = None
            else:
                if tMinus > 0:
                    validRange[0] = tMinus+epsilon
                if tPlus < 1:
                    validRange[1] = tPlus-epsilon

        return validRange

    def intersection(self, position, direction, maxDistance) -> (bool, float, Vector):
        wasBelow = None
        startingPointOnSurface = self.z(position.x, position.y)
        if startingPointOnSurface is not None:
            wasBelow = position.z < startingPointOnSurface

        if wasBelow is None or wasBelow is False:
            return (False, None, None)
        current = Vector(position)  # Copy
        delta = 0.1
        t = 0
        while abs(delta) > 0.0000001:
            t += delta
            current = position + t * direction * maxDistance
            surfaceZ = self.z(current.x, current.y)
            isBelow = None
            if surfaceZ is not None:
                isBelow = (current.z < surfaceZ)

            if wasBelow is None or isBelow is None:
                pass
            elif wasBelow != isBelow:
                delta = -delta * 0.5

            # print(position, current, wasBelow, isBelow, t)
            wasBelow = isBelow

        if t >= 0 and t <= 1.0:
            return (True, t * maxDistance, current)
        else:
            return (False, None, None)

    def smallestValidT(self, tm, tp):
        valid = []
        if tm >= 0 and tm <= 1:
            valid.append(tm)
        if tp >= 0 and tp <= 1:
            valid.append(tp)

        if len(valid) == 0:
            return None
        elif len(valid) == 1:
            return valid[0]
        elif len(valid) == 2:
            return min(valid)


    def z(self, x, y):
        """ This z represents the surface of the interface 
        as a function of r, the distance from the axis.  

        Obtained from https://en.wikipedia.org/wiki/Aspheric_lens

        """
        r = np.sqrt(x*x+y*y)
        z = r*r/(self.R*(1+sqrt(1-(1+self.kappa)*r*r/self.R/self.R)))   
        return z

    def dzdr(self, r):
        """ This approximates the slope of the surface which 
        we can then use to calculate the tangent or normal 
        to the surface.

        An analytical expression is possible and should be derived.
        """

        if self.z(r) is None:
            return None, None

        dy1 = 0.000001
        dy2 = 0.000001
        z1 = self.z(r+dy1)
        if z1 is None:
            dy1 = 0
            z1 = self.z(r) 

        z2 = self.z(r-dy2) 
        if z2 is None:
            dy2 = 0
            z2 = self.z(r) 

        dz = z1-z2
        return dz, dy1+dy2

    def surfaceNormal(self, r):
        """ Returns the surface normal at r, pointing backward.
        (between -pi/2 and pi/2).
        The angle is measured with respect to the optical axis
        (i.e. the propagation axis).  For incidence on a convex
        interface above axis (y>0), the angle will be negative.
        """ 
        dz, dy = self.dzdr(r)
        if dz is not None:
            return -arctan2(dz, dy)        
        return None


class FresnelIntersect:
    def __init__(self, direction, surface, distance, geometry=None):
        self.surface = surface
        self.direction = direction
        self.distance = distance
        self.geometry = geometry

        # assert(self.surface.indexOutside != self.surface.indexInside)

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
