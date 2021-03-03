import numpy as np
import math
import sys
from .vector import Vector

"""
Vectors and Scalars are arrays of Vector and scalars (float, int, etc...).
They appear as list of vectors or list of scalar, they are iterable.

These classes are putting in place the structure to act on an array of values, 
possibly in parallel.  Vectors is identical to Vector with its API but it always
acts on an array of [Vector]. A possible implementation would use the GPU to perform
the operations.

This will permit expressive object-oriented code such as:

a = Vectors(N=1000)
b = Vectors(N=1000)

c = a+b

"""

class NativeVectors:
    """ This is the reference implementation of Vectors. Other classes will
    be created such as GPUVectors, NumpyVectors, CuPyVectors, and others to refine the
    implementation for speed. 
    """

    def __init__(self, vectors = None, N=None):
        self.v = []
        if vectors is not None:
            for v in vectors:
                self.v.append(Vector(v)) # always copy
        elif N is not None:
            self.v = [Vector(0,0,0)]*N
        self._iteration = 0

    @property
    def count(self):
        return len(self.v)
    
    @classmethod
    def fromScaledSum(cls, a, b, scale):
        return a.addScaled(b,scale)

    def addScaled(self, rhs, scale):
        return Vectors([v1+v2*s for (v1,v2, s) in list(zip(self.v, rhs, scale))])

    @property
    def isUnitary(self) -> [bool]:
        return [v.isUnitary for v in self.v]

    @property
    def isNull(self) -> [bool]:
        return [v.isNull for v in self.v]

    # def __repr__(self):
    #     return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)
    
    # def __str__(self):
    #     return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)

    def __len__(self):
        return len(self.v)

    def __mul__(self, scale):
        return Vectors([v1*s for (v1,s) in list(zip(self.v, scale))])

    def __rmul__(self, scale):
        return Vectors([v1*s for (v1,s) in list(zip(self.v, scale))])

    def __truediv__(self, scale):
        return Vectors([v1/s for (v1,s) in list(zip(self.v, scale))])

    def __add__(self, rhs):
        return Vectors([v1+v2 for (v1,v2) in list(zip(self.v, rhs.v))])

    def __neg__(self):
        return Vectors([-v1 for v1 in self.v])

    def __sub__(self, rhs):
        return Vectors([v1-v2 for (v1,v2) in list(zip(self.v, rhs.v))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue): 
        self.v[index] = newvalue

    def __eq__(self, rhs):
        each = [v1.isEqualTo(v2) for (v1,v2) in list(zip(self.v, rhs.v))]

        return np.array(each).all()

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < self.count:
            result = self.v[self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    def isEqualTo(self, rhs):
        return [v1.isEqualTo(v2) for (v1,v2) in list(zip(self.v, rhs.v))]

    def isAlmostEqualTo(self, rhs, epsilon):
        return [v1.isAlmostEqualTo(v2, epsilon) for (v1,v2) in list(zip(self.v, rhs.v))]

    def isParallelTo(self, rhs, epsilon=1e-7):
        return [v1.isParallelTo(v2) for (v1,v2) in list(zip(self.v, rhs.v))]

    def anyPerpendicular(self):
        return Vectors([v1.anyPerpendicular() for v1 in self.v])

    def anyUnitaryPerpendicular(self):
        return Vectors([v1.anyUnitaryPerpendicular() for v1 in self.v])

    def isInXYPlane(self, atZ, epsilon=0.001):
        return [v1.isInXYPlane(atZ=atZ, epsilon=epsilon) for v1 in self.v]

    def isInYZPlane(self, atX, epsilon=0.001):
        return [v1.isInYZPlane(atX=atX, epsilon=epsilon) for v1 in self.v]

    def isInZXPlane(self, atY, epsilon=0.001):
        return [v1.isInZXPlane(atY=atY, epsilon=epsilon) for v1 in self.v]

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001) -> bool:
        return [v1.isInPlane(origin, normal, epsilon) for v1 in self.v]

    def norm(self):
        return [v1.norm() for v1 in self.v]

    def abs(self):
        return [v1.abs() for v1 in self.v]

    def normalize(self):
        return [v1.normalize() for v1 in self.v]

    def normalized(self):
        return Vector([v1.normalized() for v1 in Vector(self)])

    def isPerpendicularTo(self, rhs, epsilon=1e-7):
        return [v1.isPerpendicularTo(v2) for (v1,v2) in list(zip(self.v, rhs.v))]

    def cross(self, rhs):
        return Vectors([v1.cross(v2) for (v1,v2) in list(zip(self.v, rhs.v))])

    def dot(self, rhs):
        return [v1.dot(v2) for (v1,v2) in list(zip(self.v, rhs.v))]

    def normalizedCrossProduct(self, rhs):
        return Vectors([v1.normalizedCrossProduct(v2) 
            for (v1,v2) in list(zip(self.v, rhs.v))])

    def normalizedDotProduct(self, rhs):
        return [v1.normalizedDotProduct(v2) for (v1,v2) in list(zip(self.v, rhs.v))]

    def angleWith(self, v, axis):
        return [v1.angleWith(v=v2,axis=v3) for (v1,v2,v3) in list(zip(self.v, v.v, axis.v))]

    def planeOfIncidence(self, normal):
        return Vectors([v1.planeOfIncidence(normal=v2) for (v1,v2) in list(zip(self.v, normal.v))])

    def angleOfIncidence(self, normal):
        dotProduct = self.dot(normal)
        correctedNormal = Vectors([n*sign(s) for (n, s) in list(zip(normal, dotProduct))])

        planeNormal = self.planeOfIncidence(correctedNormal)
        return self.angleWith(correctedNormal, axis=planeNormal), planeNormal, correctedNormal

    def rotateAround(self, u, theta):
        return Vectors([v1.rotateAround(v2,t) for (v1,v2,t) in list(zip(self.v, u.v, theta))])

# class UnitVectors(Vectors):
#     def __init__(self, x: float = 0, y: float = 0, z: float = 0):
#         Vectors.__init__(self, Vector(x, y, z).normalized())

#     def normalizedCrossProduct(self, vector):
#         if isinstance(vector, UnitVector):
#             return self.cross(vector)
#         else:
#             return Vector.normalizedCrossProduct(self, vector)

#     def normalizedDotProduct(self, vector):
#         if isinstance(vector, UnitVector):
#             return self.dot(vector)
#         else:
#             return Vector.normalizedDotProduct(self, vector)


# class ConstVectors(Vectors):
#     def __init__(self, x: float = 0, y: float = 0, z: float = 0):
#         self._abs = 0
#         self._norm = 0
#         Vector.__init__(self, x, y, z)
#         self._abs = self.abs()
#         self._norm = self.norm()

#     def normalize(self):
#         if self._norm != 1:
#             raise RuntimeError("You cannot normalize a constant vector: you can use Vector instead.")
#         else:
#             raise RuntimeError("You cannot normalize a constant vector: you can use ConstUnitVector instead for unit vectors, there is no need to normalize them.")

#     def norm(self):
#         return self._norm

#     def abs(self):
#         return self._abs

#     @property
#     def x(self):
#         return self._x

#     @property
#     def y(self):
#         return self._y

#     @property
#     def z(self):
#         return self._z

#     @x.setter
#     def x(self, value):
#         raise RuntimeError("You cannot change a constant vector")

#     @y.setter
#     def y(self, value):
#         raise RuntimeError("You cannot change a constant vector")

#     @z.setter
#     def z(self, value):
#         raise RuntimeError("You cannot change a constant vector")

# class ConstUnitVectors(UnitVectors):
#     def __init__(self, x: float = 0, y: float = 0, z: float = 0):
#         Vector.__init__(self, x, y, z)
#         if self.norm() != 1.0:
#             raise ValueError("Vector must be created with proper normalized values")

#     def normalize(self):
#         return self

#     def norm(self):
#         return 1.0

#     def abs(self):
#         return 1.0

#     @property
#     def x(self):
#         return self._x

#     @property
#     def y(self):
#         return self._y

#     @property
#     def z(self):
#         return self._z

#     @x.setter
#     def x(self, value):
#         raise RuntimeError("You cannot change a constant vector")

#     @y.setter
#     def y(self, value):
#         raise RuntimeError("You cannot change a constant vector")

#     @z.setter
#     def z(self, value):
#         raise RuntimeError("You cannot change a constant vector")

# oHat = ConstVector(0, 0, 0)    
# xHat = ConstUnitVector(1, 0, 0)
# yHat = ConstUnitVector(0, 1, 0)
# zHat = ConstUnitVector(0, 0, 1)

Vectors = NativeVectors
