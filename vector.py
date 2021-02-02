import numpy as np
import math
from collections import namedtuple

class Vector:
    def __init__(self, x:float=0,y:float=0,z:float=0):
        if isinstance(x, (int, float)):
            self.x = x
            self.y = y 
            self.z = z
        elif isinstance(x, Vector):
            self.x = x.x
            self.y = x.y 
            self.z = x.z 
        elif isinstance(x, np.ndarray):
            self.x = x
            self.y = y 
            self.z = z
        else:
            raise ValueError("No valid input for Vector")

    @classmethod
    def fromScaledSum(a, b, scale):
        return Vector(a.x + b.x*scale, a.y + b.y*scale, a.z + b.z*scale)

    @property
    def isUnitary(self) -> bool:
        return abs(self.norm()-1)<1e-7

    def __repr__(self):
        return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)
    
    def __str__(self):
        return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)

    def __mul__(self, scale):
        return Vector(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale):
        return Vector(self.x * scale, self.y * scale, self.z * scale)

    def __div__(self, scale):
        return self.v / scale

    def __add__(self, vector):
        return Vector(self.x + vector.x, self.y + vector.y, self.z + vector.z)

    def __radd__(self, vector):
        return Vector(self.x + vector.x, self.y + vector.y, self.z + vector.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def addScaled(self, vector, scale):
        self.x += vector.x*scale
        self.y += vector.y*scale
        self.z += vector.z*scale

    def __sub__(self, vector):
        return Vector(self.x - vector.x, self.y - vector.y, self.z - vector.z)

    def __rsub__(self, vector):
        return Vector(-self.x + vector.x, -self.y + vector.y, -self.z + vector.z)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise ValueError("Out of range index: must be 0,1 or 2")

    def isParallelTo(self, vector):
        return (self.normalizedDotProduct(vector) - 1 < 1e-6)

    def isPerpendicularTo(self, vector):
        return (self.normalizedDotProduct(vector) < 1e-6)

    def isInXYPlane(self, atZ, epsilon=0.001) -> bool:
        if abs(self.z-z) < epsilon:
            return True
        return False

    def isInYZPlane(self, atX, epsilon=0.001) -> bool:
        if abs(self.x-x) < epsilon:
            return True
        return False

    def isInZXPlane(self, atY, epsilon=0.001) -> bool:
        if abs(self.y-y) < epsilon:
            return True
        return False

    def isInPlane(self, origin:'Vector', normal:'Vector', epsilon=0.001) -> bool:
        local = self-origin
        if abs(local.normalizedDotProduct(normal)) < epsilon:
            return True
        return False

    def norm(self):
        ux = self.x
        uy = self.y
        uz = self.z
        return ux*ux+uy*uy+uz*uz

    def abs(self):
        ux = self.x
        uy = self.y
        uz = self.z
        return math.sqrt(ux*ux+uy*uy+uz*uz)

    def normalize(self):
        length = self.abs()
        self.x /= length
        self.y /= length
        self.z /= length

    def cross(self, vector):
        """ Accessing properties is costly when done very often.
        cross product is a common operation """
        ux = self.x
        uy = self.y
        uz = self.z
        vx = vector.x
        vy = vector.y
        vz = vector.z
        return Vector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)

    def dot(self, vector):
        return self.x*vector.x + self.y*vector.y + self.z*vector.z 

    def normalizedCrossProduct(self, vector):
        productNorm = self.norm() * vector.norm()
        if productNorm == 0:
            return 0
        return self.cross(vector) * (1 / math.sqrt(productNorm))

    def normalizedDotProduct(self, vector):
        productNorm = self.norm() * vector.norm()
        if productNorm == 0:
            return 0
        return self.dot(vector) * (1 / math.sqrt(productNorm))

    def angleWith(self, v, righthand):
        sinPhi = self.normalizedCrossProduct(v)
        sinPhiAbs = sinPhi.abs()
        phi = math.asin(sinPhiAbs)
    
        if self.dot(v) <= 0:
            phi = math.pi-phi

        if sinPhi.dot(righthand) <= 0:
            phi = -phi
    
        return phi

    def rotateAround(self, u, theta):
        # This is the most expensive (and most common)
        # operation when performing Monte Carlo in tissue 
        # (40% of time spent here). It is difficult to optimize without
        # making it even less readable than it currently is
        # http://en.wikipedia.org/wiki/Rotation_matrix
        # 
        # Several options were tried in the past such as 
        # external not-so-portable C library, unreadable 
        # shortcuts, sine and cosine lookup tables, etc... 
        # and the performance gain was minimal (<20%).
        # For now, this is the best, most readable solution.

        u.normalize()

        cost = math.cos(theta)
        sint = math.sin(theta)
        one_cost = 1 - cost
        
        ux = u.x
        uy = u.y
        uz = u.z
        
        X = self.x
        Y = self.y
        Z = self.z
        
        self.x = (cost     + ux*ux    * one_cost ) * X \
        +        (ux*uy    * one_cost - uz * sint) * Y \
        +        (ux * uz  * one_cost + uy * sint) * Z
        self.y = (uy*ux    * one_cost + uz * sint) * X \
        +        (cost     + uy*uy    * one_cost ) * Y \
        +        (uy * uz  * one_cost - ux * sint) * Z
        self.z = (uz*ux    * one_cost - uy * sint) * X \
        +        (uz * uy  * one_cost + ux * sint) * Y \
        +        (cost     + uz*uz    * one_cost ) * Z

    def rotateAroundX(self, phi):
        v = Vector(self.x, self.y, self.z)
        
        c = math.cos(phi);
        s = math.sin(phi);
    
        self.y = c * v.y - s * v.z;
        self.z = s * v.y + c * v.z;

    def rotateAroundY(self, phi):
        v = Vector(self.x, self.y, self.z)
        
        c = math.cos(phi)
        s = math.sin(phi)
    
        self.x = c * v.x + s * v.z
        self.z = -s * v.x + c * v.z

    def rotateAroundZ(self, phi):
        v = Vector(self.x, self.y, self.z)
        
        c = math.cos(phi)
        s = math.sin(phi)
    
        self.x = c * v.x - s * v.y
        self.y = s * v.x + c * v.y
        self.z = v.z

class UnitVector(Vector):
    def __init__(self, x:float=0,y:float=0,z:float=0):
        Vector.__init__(self, x,y,z)

    def abs(self):
        """ The `sqrt()` calculation normally used to compute `Vector.abs()`
        is expensive. If a vector should be unitary in the first place, 
        then we can use sqrt(1+x) = 1+x/2+...
        with norm = 1 + x, or norm - 1 = x. Sneaky, but efficient.
        We still need to compute `abs()` because we want to normalize
        the vector manually after calculations because if we don't,
        round-off errors will take over."""
        ux = self.x
        uy = self.y
        uz = self.z
        return (ux*ux+uy*uy+uz*uz+1)/2

    def cross(self, vector):
        """ Accessing properties is costly when done very often.
        cross product of unit vectors is a common operation """
        ux = self.x
        uy = self.y
        uz = self.z
        vx = vector.x
        vy = vector.y
        vz = vector.z
        if isinstance(vector, UnitVector):
            return UnitVector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)
        else:
            return Vector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)

    def normalizedCrossProduct(self, vector):
        if isinstance(vector, UnitVector):
            return self.cross(vector)
        else:
            return Vector.normalizedCrossProduct(self, vector)

    def normalizedDotProduct(self, vector):
        if isinstance(vector, UnitVector):
            return self.dot(vector)
        else:
            return Vector.normalizedDotProduct(self, vector)

xHat = UnitVector(1,0,0)
yHat = UnitVector(0,1,0)
zHat = UnitVector(0,0,1)

