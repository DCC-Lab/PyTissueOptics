import numpy as np
import math
import sys
import random
from .vector import Vector
from .scalars import *

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

    def __init__(self, vectors=None, N=None):
        self.v = []
        if vectors is not None:
            for v in vectors:
                self.v.append(Vector(v))  # always copy
        elif N is not None:
            self.v = [Vector(0, 0, 0)] * N
        self._iteration = 0

    @property
    def count(self):
        return len(self.v)

    @classmethod
    def random(cls, N):
        vectors = []
        for i in range(N):
            x = random.random() * 2 - 1
            y = random.random() * 2 - 1
            z = random.random() * 2 - 1
            vectors.append(Vector(x, y, z))
        return Vectors(vectors)

    @classmethod
    def randomUnitary(cls, N):
        vectors = []
        for i in range(N):
            x = random.random() * 2 - 1
            y = random.random() * 2 - 1
            z = random.random() * 2 - 1
            vectors.append(Vector(x, y, z).normalized())
        return Vectors(vectors)

    @classmethod
    def fromScaledSum(cls, a, b, scale):
        return a.addScaled(b, scale)

    def addScaled(self, rhs, scale):
        return Vectors([v1 + v2 * s for (v1, v2, s) in list(zip(self.v, rhs, scale))])

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
        return Vectors([v1 * s for (v1, s) in list(zip(self.v, scale))])

    def __rmul__(self, scale):
        return Vectors([v1 * s for (v1, s) in list(zip(self.v, scale))])

    def __truediv__(self, scale):
        return Vectors([v1 / s for (v1, s) in list(zip(self.v, scale))])

    def __add__(self, rhs):
        return Vectors([v1 + v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __neg__(self):
        return Vectors([-v1 for v1 in self.v])

    def __sub__(self, rhs):
        return Vectors([v1 - v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue):
        self.v[index] = newvalue

    def __eq__(self, rhs):
        each = [v1.isEqualTo(v2) for (v1, v2) in list(zip(self.v, rhs.v))]
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
        return Scalars([v1.isEqualTo(v2) for (v1, v2) in list(zip(self.v, rhs))])

    def isAlmostEqualTo(self, rhs, epsilon):
        return Scalars([v1.isAlmostEqualTo(v2, epsilon) for (v1,v2) in list(zip(self.v, rhs))])

    def isParallelTo(self, rhs, epsilon=1e-7):
        return Scalars([v1.isParallelTo(v2) for (v1, v2) in list(zip(self.v, rhs))])

    def anyPerpendicular(self):
        return Vectors([v1.anyPerpendicular() for v1 in self.v])

    def anyUnitaryPerpendicular(self):
        return Vectors([v1.anyUnitaryPerpendicular() for v1 in self.v])

    def isInXYPlane(self, atZ, epsilon=0.001):
        return Scalars([v1.isInXYPlane(atZ=atZ, epsilon=epsilon) for v1 in self.v])

    def isInYZPlane(self, atX, epsilon=0.001):
        return Scalars([v1.isInYZPlane(atX=atX, epsilon=epsilon) for v1 in self.v])

    def isInZXPlane(self, atY, epsilon=0.001):
        return Scalars([v1.isInZXPlane(atY=atY, epsilon=epsilon) for v1 in self.v])

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001) -> bool:
        return Scalars([v1.isInPlane(origin, normal, epsilon) for v1 in self.v])

    def norm(self):
        return Scalars([v1.norm() for v1 in self.v])

    def abs(self):
        return Scalars([v1.abs() for v1 in self.v])

    def normalize(self):
        [v1.normalize() for v1 in self.v]
        return self

    def normalized(self):
        return Vectors([v1.normalized() for v1 in self.v])

    def isPerpendicularTo(self, rhs, epsilon=1e-7):
        return Scalars([v1.isPerpendicularTo(v2) for (v1, v2) in list(zip(self.v, rhs))])

    def cross(self, rhs):
        return Vectors([v1.cross(v2) for (v1, v2) in list(zip(self.v, rhs))])

    def dot(self, rhs):
        return Scalars([v1.dot(v2) for (v1, v2) in list(zip(self.v, rhs))])

    def normalizedCrossProduct(self, rhs):
        return Vectors([v1.normalizedCrossProduct(v2) 
            for (v1,v2) in list(zip(self.v, rhs))])

    def normalizedDotProduct(self, rhs):
        return Scalars([v1.normalizedDotProduct(v2) for (v1, v2) in list(zip(self.v, rhs))])

    def angleWith(self, v, axis):
        return Scalars([v1.angleWith(v=v2, axis=v3) for (v1, v2, v3) in list(zip(self.v, v, axis))])

    def planeOfIncidence(self, normal):
        return Vectors([v1.planeOfIncidence(normal=v2) for (v1, v2) in list(zip(self.v, normal))])

    def angleOfIncidence(self, normal):
        dotProduct = self.dot(normal)
        correctedNormal = Vectors([n*(1.0 - 2.0 * (s < 0)) for (n, s) in list(zip(normal, dotProduct.v))])

        planeNormal = self.planeOfIncidence(correctedNormal)
        angles = Scalars(self.angleWith(correctedNormal, axis=planeNormal))
        return angles, planeNormal, correctedNormal

    def rotateAround(self, u, theta):
        [v1.rotateAround(v2, t) for (v1, v2, t) in list(zip(self.v, u.v, theta))]
        return self

    def rotatedAround(self, u, theta):
        v = Vectors(self) # copy
        [v1.rotateAround(v2,t) for (v1,v2,t) in list(zip(v.v, u.v, theta))]
        return v

class NumpyVectors:
    """ This is the Reference Vectors Class for numpy-like calculations.
    This architecture will be used by cupy since it is a drop-in replacement
    """

    def __init__(self, vectors=None, N=None):
        if vectors is not None:
            self.v = np.asarray(vectors)
        elif N is not None:
            self.v = np.zeros(3, N, dtype=np.float32)
            
        self._iteration = 0
    
    def __len__(self):
        return self.v.shape[1]

    def __mul__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.multiply(self.v, other.v))
        elif isinstance(other, NumpyScalars):
            return NumpyVectors(np.multiply(self.v * other.v[:, None]))
        else:
            return NumpyVectors(np.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.true_div(self.v, other.v))
        elif isinstance(other, NumpyScalars):
            return NumpyVectors(np.true_div(self.v * other.v[:, None]))
        else:
            return NumpyVectors(np.true_div(self.v, other))

    def __add__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.add(self.v, other.v))
        else:
            return NumpyVectors(np.add(self.v, other))

    def __sub__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.subtract(self.v, other.v))
        else:
            return NumpyVectors(np.subtract(self.v, other))

    def __neg__(self):
        return NumpyVectors(np.negative(self.v))

    def __getitem__(self, index):
        pass

    def __setitem__(self, index, value):
        pass

    def __eq__(self, rhs):
        pass

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        pass

    @property
    def isUnitary(self) -> [bool]:
        pass

    @property
    def isNull(self) -> [bool]:
        pass

    @property
    def count(self):
        return len(self.v)

    @classmethod
    def random(cls, N):
        array = np.random.rand(3, N)
        return array

    @classmethod
    def randomUnitary(cls, N):
        pass

    def isEqualTo(self, rhs):
        pass

    def isAlmostEqualTo(self, rhs, epsilon):
        pass

    def isParallelTo(self, rhs, epsilon=1e-7):
        pass

    def anyPerpendicular(self):
        pass

    def anyUnitaryPerpendicular(self):
        pass

    def isInXYPlane(self, atZ, epsilon=0.001):
        pass

    def isInYZPlane(self, atX, epsilon=0.001):
        pass

    def isInZXPlane(self, atY, epsilon=0.001):
        pass

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001) -> bool:
        pass

    def norm(self):
        pass

    def abs(self):
        pass

    def normalize(self):
        pass
        return self

    def normalized(self):
        pass

    def isPerpendicularTo(self, rhs, epsilon=1e-7):
        pass

    def cross(self, rhs):
        pass

    def dot(self, rhs):
        pass

    def normalizedCrossProduct(self, rhs):
        pass

    def normalizedDotProduct(self, rhs):
        pass

    def angleWith(self, v, axis):
        pass

    def planeOfIncidence(self, normal):
        pass

    def angleOfIncidence(self, normal):
        pass

    def rotateAround(self, u, theta):
        pass

Vectors = NativeVectors
