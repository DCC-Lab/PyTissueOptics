import numpy as np
import math
import sys
import random
from .vector import Vector, oHat
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
                if v is not None:
                    self.v.append(Vector(v))  # always copy
                else:
                    self.v.append(None)
        elif N is not None:
            self.v = [Vector(0, 0, 0)] * N
        self._iteration = 0
        self.selected = Scalars([True]*len(self.v))

    # def selectedAll(self):
    #     self.selected = Scalars([True] * len(self.v))
    #
    def selectedVectors(self):
        return Vectors([v1 if e else None for (v1, e) in list(zip(self.v, self.selected))])

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
        return Vectors([v1 if not e else v1 + v2 * s for (v1, v2, s, e) in list(zip(self.v, rhs, scale, self.selected))])

    @property
    def isUnitary(self) -> [bool]:
        return [ v.isUnitary if e else False for v, e in list(zip(self.v, self.selected))]

    @property
    def isNull(self) -> [bool]:
        return [v.isNull if e else False for v, e in list(zip(self.v, self.selected))]

    # def __repr__(self):
    #     return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)

    # def __str__(self):
    #     return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)

    def __len__(self):
        return len(self.v)

    def __mul__(self, scale):
        return Vectors([v1 * s if e else v1 for (v1, s, e) in list(zip(self.v, scale, self.selected))])

    def __rmul__(self, scale):
        return Vectors([v1 * s if e else v1 for (v1, s, e) in list(zip(self.v, scale, self.selected))])

    def __truediv__(self, scale):
        return Vectors([v1 / s if e else v1 for (v1, s, e) in list(zip(self.v, scale, self.selected))])

    def __add__(self, rhs):
        return Vectors([v1 + v2 if e else v1 for (v1, v2, e) in list(zip(self.v, rhs.v, self.selected))])

    def __neg__(self):
        return Vectors([-v1 if e else v1 for v1, e in list(zip(self.v, self.selected))])

    def __sub__(self, rhs):
        return Vectors([v1 - v2 if e else v1 for (v1, v2, e) in list(zip(self.v, rhs.v, self.selected))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue):
        self.v[index] = newvalue

    def __eq__(self, rhs):
        each = [ v1 == v2 if e else False for (v1, v2, e) in list(zip(self.v, rhs, self.selected))]
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
        return Scalars([False if not e or v1 is None or v2 is None else v1.isEqualTo(v2) for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def isAlmostEqualTo(self, rhs, epsilon):
        return Scalars([False if not e or v1 is None or v2 is None else v1.isAlmostEqualTo(v2, epsilon) for (v1,v2,e) in list(zip(self.v, rhs, self.selected))])

    def isParallelTo(self, rhs, epsilon=1e-7):
        return Scalars([False if not e or v1 is None else v1.isParallelTo(v2) for (v1, v2,e) in list(zip(self.v, rhs, self.selected))])

    def anyPerpendicular(self):
        return Vectors([oHat if not e or v1 is None else v1.anyPerpendicular() for v1,e in list(zip(self.v, self.selected))])

    def anyUnitaryPerpendicular(self):
        return Vectors([oHat if not e or v1 is None else v1.anyUnitaryPerpendicular() for v1,e in list(zip(self.v, self.selected))])

    def isInXYPlane(self, atZ, epsilon=0.001):
        return Scalars([v1.isInXYPlane(atZ=atZ, epsilon=epsilon) if e else False for v1,e in list(zip(self.v, self.selected))])

    def isInYZPlane(self, atX, epsilon=0.001):
        return Scalars([v1.isInYZPlane(atX=atX, epsilon=epsilon) if e else False for v1,e in list(zip(self.v, self.selected))])

    def isInZXPlane(self, atY, epsilon=0.001):
        return Scalars([v1.isInZXPlane(atY=atY, epsilon=epsilon) if e else False for v1,e in list(zip(self.v, self.selected))])

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001) -> bool:
        return Scalars([v1.isInPlane(origin, normal, epsilon) if e else False for v1,e in list(zip(self.v, self.selected))])

    def norm(self):
        return Scalars([v1.norm() if e else 0 for v1,e in list(zip(self.v, self.selected))])

    def abs(self):
        return Scalars([v1.abs() if e else 0 for v1,e in list(zip(self.v, self.selected))])

    def normalize(self):
        [v1.normalize() if e else v1 for v1,e in list(zip(self.v, self.selected))]
        return self

    def normalized(self):
        return Vectors([v1.normalized() if e else v1 for v1,e in list(zip(self.v, self.selected))])

    def isPerpendicularTo(self, rhs, epsilon=1e-7):
        return Scalars([v1.isPerpendicularTo(v2) if e else False for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def cross(self, rhs):
        return Vectors([v1.cross(v2) if e else v1 for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def dot(self, rhs):
        return Scalars([v1.dot(v2) if e else 0 for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def normalizedCrossProduct(self, rhs):
        return Vectors([v1.normalizedCrossProduct(v2) if e else v1
            for (v1,v2,e) in list(zip(self.v, rhs, self.selected))])

    def normalizedDotProduct(self, rhs):
        return Scalars([v1.normalizedDotProduct(v2) if e else 0 for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def angleWith(self, v, axis):
        return Scalars([v1.angleWith(v=v2, axis=v3) if e else 0 for (v1, v2, v3, e) in list(zip(self.v, v, axis, self.selected))])

    def planeOfIncidence(self, normal):
        return Vectors([v1.planeOfIncidence(normal=v2) if e else oHat for (v1, v2, e) in list(zip(self.v, normal, self.selected))])

    def angleOfIncidence(self, normal):
        dotProduct = self.dot(normal)
        normal.selected = (dotProduct.v < 0)
        correctedNormal = -normal

        planeNormal = self.planeOfIncidence(correctedNormal)
        angles = Scalars(self.angleWith(correctedNormal, axis=planeNormal))
        return angles, planeNormal, correctedNormal

    def rotateAround(self, u, theta):
        [v1.rotateAround(v2, t) if e else v1 for (v1, v2, t, e) in list(zip(self.v, u.v, theta, self.selected))]
        return self

    def rotatedAround(self, u, theta):
        v = Vectors(self) # copy
        [v1.rotateAround(v2,t) if e else v1 for (v1,v2,t, e) in list(zip(v.v, u.v, theta, self.selected))]
        return v


class NumpyVectors:
    """ This is the Reference Vectors Class for numpy-like calculations.
    This architecture will be used by cupy since it is a drop-in replacement
    """

    def __init__(self, vectors=None, N=None):
        if vectors is not None:
            if type(vectors) == np.ndarray:
                self.v = vectors.astype('float64')
            else:
                self.v = np.asarray(vectors, dtype=np.float64)
        elif N is not None:
            self.v = np.zeros((N, 3), dtype=np.float64)
            
        self._iteration = 0
    
    def __len__(self):
        return self.v.shape[0]

    def __mul__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.multiply(self.v, other.v))
        elif isinstance(other, NumpyScalars):
            return NumpyVectors(np.multiply(self.v, other.v[:, None]))
        # elif isinstance(other, np.ndarray):
        #     if len(other.shape) == 1:
        #         return NumpyVectors(self.v * other[:, None])
        else:
            return NumpyVectors(np.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.true_divide(self.v, other.v))
        elif isinstance(other, NumpyScalars):
            return NumpyVectors(np.true_divide(self.v * other.v[:, None]))
        else:
            return NumpyVectors(np.true_divide(self.v, other))

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

    def __eq__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.equal(self.v, other.v))
        else:
            return NumpyVectors(np.subtract(self.v, other))

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return str(self.v)

    """ The getitem, setitem, iter, next special methods should not be used
    because never should there be need to bypass the numpy function. Such use
    could and will deteriorate performances and possibly fail to parallelize.
    Can be used to unit test """

    def __getitem__(self, index):
        return self.v[index, :]

    def __setitem__(self, index, value):
        self.v[index, :] = value

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        result = self.v[:, self._iteration]
        self._iteration += 1
        return result

    @property
    def isUnitary(self):
        return np.less(np.abs(np.linalg.norm(self.v, axis=1))-1, 1e-9)

    @property
    def isNull(self):
        return np.less(np.linalg.norm(self.v, axis=1), 1e-9)

    @property
    def count(self):
        return len(self.v)

    @classmethod
    def randomUniform(cls, N, r):
        theta = (np.random.rand(N) * 2 * np.pi)
        phi = (np.random.rand(N) * np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return NumpyVectors(np.stack((x, y, z), axis=-1))

    @classmethod
    def randomUniformUnitary(cls, N):
        theta = np.random.rand(N) * 2 * np.pi
        phi = np.random.rand(N) * np.pi
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        return NumpyVectors(np.stack((x, y, z), axis=-1))

    def isEqualTo(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other.v)), 1e-9))
        else:
            return NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other)), 1e-9))

    def isAlmostEqualTo(self, other, epsilon):
        if isinstance(other, NumpyVectors):
            return NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other.v)), epsilon))
        else:
            return NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other)), epsilon))

    def isParallelTo(self, other, epsilon=1e-9):
        return np.less(self.normalizedDotProduct(other.v) - 1, epsilon)

    def isPerpendicularTo(self, other, epsilon=1e-9):
        return np.less(self.normalizedDotProduct(other.v), epsilon)

    def anyPerpendicular(self):
        # check if x or y is zero, if yes, cross yHat elif z is 0: set to none, if not, cross with xHat
        xHat = np.array([1, 0, 0]).transpose()
        yHat = np.array([0, 1, 0]).transpose()
        convA = np.array([1, 1, 0]).transpose()
        convB = np.array([0, 0, 1]).transpose()
        YZ0 = self * convA
        Z0 = self * convB
        P = np.all(YZ0.v == 0, axis=1)
        Q = np.all(Z0.v == 0, axis=1)
        not_P = np.logical_not(P)
        P_and_Q = np.logical_and(P, Q)
        maskXY0 = np.logical_xor(P, P_and_Q)
        maskXYZ0 = P_and_Q
        maskXY1 = not_P

        aXY0 = self * maskXY0[:, None]
        aXYZ0 = self * maskXYZ0[:, None]
        aXY1 = self * maskXY1[:, None]

        aXY0Cross = aXY0.cross(xHat)
        aXY1Cross = aXY1.cross(yHat)

        output = aXY0Cross + aXY1Cross

        # print("X=0, Y=0 :\n{}\n".format(aXY0))
        # print("X!=0, Y!=0 ; \n{}\n".format(aXY1))
        #
        # print("X=0, Y=0 :\n{}\n".format(aXY0Cross))
        # print("X!=0, Y!=0 ; \n{}\n".format(aXY1Cross))
        #
        # print("OUTPUT:\n{}\n".format(output))

        return output

    def anyUnitaryPerpendicular(self):
        pass

    def isInXYPlane(self, atZ, epsilon=0.001):
        pass

    def isInYZPlane(self, atX, epsilon=0.001):
        pass

    def isInZXPlane(self, atY, epsilon=0.001):
        pass

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001):
        pass

    def norm(self):
        return NumpyScalars(np.linalg.norm(self.v, axis=1))

    def abs(self):
        return NumpyVectors(np.abs(self.v))

    def normalize(self):
        self.v = self.v/np.linalg.norm(self.v, axis=1)

    def normalized(self):
        return NumpyVectors(self.v/np.linalg.norm(self.v, axis=1))

    def cross(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.cross(self.v, other.v))
        else:
            return NumpyVectors(np.cross(self.v, other))

    def dot(self, other):
        # element-wise dot product(fake np.dot)
        # https://stackoverflow.com/questions/41443444/numpy-element-wise-dot-product
        if isinstance(other, NumpyVectors):
            return NumpyScalars(np.einsum('ij,ij->i', self.v, other.v))
        else:
            pass

    def normalizedCrossProduct(self, other):
        '''TODO:  Is this OK'''
        return self.cross(other).normalize()

    def normalizedDotProduct(self, other):
        '''TODO:  find way to calculate the zeors'''
        norm = self.norm() * other.norm()
        return self.dot(other) * norm * np.exp(-0.5)

    def angleWith(self, v, axis):
        pass

    def planeOfIncidence(self, normal):
        pass

    def angleOfIncidence(self, normal):
        pass

    def rotateAround(self, u, theta):
        pass


Vectors = NativeVectors
