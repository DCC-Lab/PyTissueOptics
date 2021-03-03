import numpy as np
import math
import sys
import random
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

    def __init__(self, vectors=None, N=None):
        self.v = []
        if vectors is not None:
            for v in vectors:
                self.v.append(Vector(v)) # always copy
        elif N is not None:
            self.v = [Vector(0, 0, 0)]*N
        self._iteration = 0

    @property
    def count(self):
        return len(self.v)

    @classmethod
    def random(cls, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( Vector(x,y,z) )
        return Vectors(vectors)

    @classmethod
    def randomUnitary(cls, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( Vector(x,y,z).normalized() )
        return Vectors(vectors)

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
        [v1.normalize() for v1 in self.v]
        return self

    def normalized(self):
        return Vectors([v1.normalized() for v1 in self.v])

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
        correctedNormal = Vectors([n*(1-2*(s<0)) for (n, s) in list(zip(normal, dotProduct))])

        planeNormal = self.planeOfIncidence(correctedNormal)
        return self.angleWith(correctedNormal, axis=planeNormal), planeNormal, correctedNormal

    def rotateAround(self, u, theta):
        [v1.rotateAround(v2,t) for (v1,v2,t) in list(zip(self.v, u.v, theta))]
        return self

    def rotatedAround(self, u, theta):
        v = Vectors(self) # copy
        [v1.rotateAround(v2,t) for (v1,v2,t) in list(zip(v.v, u.v, theta))]
        return v

class NumpyVectors:
    """ This is the Numpy implementation of Vectors using numpy arrays
    """

    def __init__(self, count: int):
        self.v = np.ndarray(3, count, dtype=np.float32)
        self._iteration = 0

    def __len__(self):
        return self.v.shape[1]

    def __mul__(self, val):
        pass

    def __rmul__(self, scale):
        pass

    def __truediv__(self, scale):
        pass

    def __add__(self, rhs):
        pass

    def __neg__(self):
        pass

    def __sub__(self, rhs):
        pass

    def __getitem__(self, index):
        pass

    def __setitem__(self, index, newvalue):
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

Vectors = NativeVectors
