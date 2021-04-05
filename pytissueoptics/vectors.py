import numpy as np
try:
    import cupy as cp
except:
    cp = np

import random
from .vector import Vector, oHat
from .scalars import *
import copy

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

    def selectAll(self):
        self.selected = Scalars([True] * len(self.v))

    def select(self, which):
        self.selected = Scalars(which)

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

    def replaceSelected(self, v, selected=None):
        if selected is None:
            selected = self.selected

        for i in range(len(self.v)):
            if selected[i]:
                self.v[i] = v[i]

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

    def isPerpendicularTo(self, rhs, epsilon=1e-7):
        return Scalars([False if not e or v1 is None else v1.isPerpendicularTo(v2) for (v1, v2,e) in list(zip(self.v, rhs, self.selected))])

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
        return Scalars([v1.normSquared() if e else 0 for v1, e in list(zip(self.v, self.selected))])

    def abs(self):
        return Scalars([v1.abs() if e else 0 for v1,e in list(zip(self.v, self.selected))])

    def normalize(self):
        [v1.normalize() if e else v1 for v1,e in list(zip(self.v, self.selected))]
        return self

    def normalized(self):
        return Vectors([v1.normalized() if e else v1 for v1,e in list(zip(self.v, self.selected))])

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
    It is the go-to fallback since numpy is available for every python user
    and since it doesn't require a GPU.
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
            return NumpyVectors(np.true_divide(self.v, other.v[:, None]))
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
        output = NumpyVectors(np.stack((x, y, z), axis=-1))
        print(output)
        return output

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

    """ TODO: Test Function """
    def isParallelTo(self, other, epsilon=1e-9):
        r = self.normalizedCrossProduct(other).norm().v
        a = np.less_equal(r, epsilon)
        r = np.where(self.isNull | other.isNull, False, a)
        return r

    """ TODO: Test Function """
    def isPerpendicularTo(self, other, epsilon=1e-9):
        r = np.abs(self.normalizedDotProduct(other).v)
        a = np.less_equal(r, epsilon)
        r = np.where(self.isNull | other.isNull, False, a)
        return r

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
        return self.anyPerpendicular().normalized()

    """ TODO: Make Function """
    def isInXYPlane(self, atZ, epsilon=0.001):
        pass

    """ TODO: Make Function """
    def isInYZPlane(self, atX, epsilon=0.001):
        pass

    """ TODO: Make Function """
    def isInZXPlane(self, atY, epsilon=0.001):
        pass

    """ TODO: Make Function """
    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001):
        pass

    def norm(self):
        return NumpyScalars(np.linalg.norm(self.v, axis=1))

    def normSquared(self):
        return NumpyScalars(self.abs)

    def abs(self):
        return NumpyVectors(np.abs(self.v))

    def normalize(self):
        """MUST verify that norm is 0."""
        norm = self.norm().v
        normNot0 = np.where(norm != 0, norm, 1)
        normalizedVectors = self.v / normNot0[:, None]
        self.v = normalizedVectors
        return self

    def normalized(self):
        """Watch out, does this modifies the self also?, yes which is why I deepcopy"""
        v = copy.deepcopy(self)
        return v.normalize()

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
            return NumpyScalars(np.einsum('ij,ij->i', self.v, other))

    def normalizedCrossProduct(self, other):
        '''TODO:  Is this OK'''
        productNorm = (self.norm() * other.norm()).v
        productNorm = np.where(productNorm != 0, productNorm, 1)
        output = self.cross(other) / productNorm[:, None]
        return output

    def normalizedDotProduct(self, other):
        '''TODO:  find way to calculate the zeors'''
        invAbs = (self.norm() * other.norm()).v
        invAbs = np.where(invAbs != 0, invAbs, 1)
        dot = self.dot(other)
        output = dot / invAbs
        return output

    def angleWith(self, v, axis):
        """ will v and axis be Vectors Array too or single vectors??"""
        sinPhi = self.normalizedCrossProduct(v)
        sinPhiAbs = sinPhi.norm()
        phi = np.arcsin(sinPhiAbs.v)
        piMinusPhi = np.pi - phi
        dotV = self.dot(v)
        dotAxis = sinPhi.dot(axis)

        phi = np.where(dotV.v <= 0, piMinusPhi, phi)
        minusPhi = -phi
        phi = np.where(dotAxis.v <= 0, minusPhi, phi)

        return NumpyScalars(phi)  # What's supposed to be the return type?

    def planeOfIncidence(self, normal):
        dotNormal = self.dot(normal)
        normal = np.where(dotNormal.v[:, None] < 0, -normal.v, normal.v)

        planeOfIncidenceNormal = self.cross(normal)
        planeNorm = planeOfIncidenceNormal.norm()

        anyPerp = self.anyPerpendicular()
        planeNormalized = planeOfIncidenceNormal.normalized()

        output = np.where(planeNorm.v[:, None] < 1e-3, anyPerp.v, planeNormalized.v)
        return NumpyVectors(output)

    def angleOfIncidence(self, normal):
        dotNormal = self.dot(normal)
        normal = NumpyVectors(np.where(dotNormal.v[:, None] < 0, -normal.v, normal.v))

        planeNormal = self.planeOfIncidence(normal)
        return self.angleWith(normal, axis=planeNormal), planeNormal, normal

    """ TODO: Test Function """
    def rotateAround(self, u, theta):
        u.normalize()
        #print(theta.v)
        cost = (np.cos(theta.v))
        sint = (np.sin(theta.v))
        one_cost = (1 - cost)

        ux = u.v[:, 0]
        uy = u.v[:, 1]
        uz = u.v[:, 2]

        X = self.v[:, 0]
        Y = self.v[:, 1]
        Z = self.v[:, 2]

        x = (cost + ux * ux * one_cost) * X + (ux * uy * one_cost - uz * sint) * Y + (ux * uz * one_cost + uy * sint) * Z
        y = (uy * ux * one_cost + uz * sint) * X + (cost + uy * uy * one_cost) * Y + (uy * uz * one_cost - ux * sint) * Z
        z = (uz * ux * one_cost - uy * sint) * X + (uz * uy * one_cost + ux * sint) * Y + (cost + uz * uz * one_cost) * Z

        self.v = np.stack((x, y, z), axis=-1)

        return self


class CupyVectors:
    """ This is the Reference Cupy Vectors Class for cupy-like calculations.
    This architecture will be used to do GPU calculations on hosts
    that have CUDA libraries and cupy. This requires a NVIDIA GPU. Will fallback
    to OpenCLVectors or NumpyVectors depending on requirements met.
    """

    def __init__(self, vectors=None, N=None):
        if vectors is not None:
            if type(vectors) == cp.ndarray:
                self.v = vectors.astype('float64')

            else:
                self.v = cp.asarray(vectors, dtype=cp.float64)
        elif N is not None:
            self.v = cp.zeros((N, 3), dtype=cp.float64)

        self._iteration = 0

    def __len__(self):
        return self.v.shape[0]

    def __mul__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.multiply(self.v, other.v))
        elif isinstance(other, CupyScalars):
            return CupyVectors(cp.multiply(self.v, other.v[0][:, None]))
        # elif isinstance(other, cp.ndarray):
        #     if len(other.shape) == 1:
        #         return CupyVectors(self.v * other[:, None])
        else:
            return CupyVectors(cp.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.true_divide(self.v, other.v))
        elif isinstance(other, CupyScalars):
            return CupyVectors(cp.true_divide(self.v, other.v[:, None]))
        else:
            return CupyVectors(cp.true_divide(self.v, other))

    def __add__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.add(self.v, other.v))
        else:
            return CupyVectors(cp.add(self.v, other))

    def __sub__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.subtract(self.v, other.v))
        else:
            return CupyVectors(cp.subtract(self.v, other))

    def __neg__(self):
        return CupyVectors(cp.negative(self.v))

    def __eq__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.equal(self.v, other.v))
        else:
            return CupyVectors(cp.subtract(self.v, other))

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

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        result = self.v[:, self._iteration]
        self._iteration += 1
        return result

    @property
    def isUnitary(self):
        return cp.less(cp.abs(cp.linalg.norm(self.v, axis=1)) - 1, 1e-9)

    @property
    def isNull(self):
        return cp.less(cp.linalg.norm(self.v, axis=1), 1e-9)

    @property
    def count(self):
        return len(self.v)

    @classmethod
    def randomUniform(cls, N, r):
        theta = (cp.random.rand(N) * 2 * cp.pi)
        phi = (cp.random.rand(N) * cp.pi)
        x = r * cp.sin(phi) * cp.cos(theta)
        y = r * cp.sin(phi) * cp.sin(theta)
        z = r * cp.cos(phi)
        return CupyVectors(cp.stack((x, y, z), axis=-1))

    @classmethod
    def randomUniformUnitary(cls, N):
        theta = cp.random.rand(N) * 2 * cp.pi
        phi = cp.random.rand(N) * cp.pi
        x = cp.sin(phi) * cp.cos(theta)
        y = cp.sin(phi) * cp.sin(theta)
        z = cp.cos(phi)
        output = CupyVectors(cp.stack((x, y, z), axis=-1))
        print(output)
        return output

    def isEqualTo(self, other):
        if isinstance(other, CupyVectors):
            return CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other.v)), 1e-9))
        else:
            return CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other)), 1e-9))

    def isAlmostEqualTo(self, other, epsilon):
        if isinstance(other, CupyVectors):
            return CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other.v)), epsilon))
        else:
            return CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other)), epsilon))

    """ TODO: Test Function """

    def isParallelTo(self, other, epsilon=1e-9):
        return cp.less(self.normalizedDotProduct(other.v) - 1, epsilon)

    """ TODO: Test Function """

    def isPerpendicularTo(self, other, epsilon=1e-9):
        return cp.less(self.normalizedDotProduct(other.v), epsilon)

    def anyPerpendicular(self):
        # check if x or y is zero, if yes, cross yHat elif z is 0: set to none, if not, cross with xHat
        xHat = cp.array([1, 0, 0]).transpose()
        yHat = cp.array([0, 1, 0]).transpose()
        convA = cp.array([1, 1, 0]).transpose()
        convB = cp.array([0, 0, 1]).transpose()
        YZ0 = self * convA
        Z0 = self * convB
        P = cp.all(YZ0.v == 0, axis=1)
        Q = cp.all(Z0.v == 0, axis=1)
        not_P = cp.logical_not(P)
        P_and_Q = cp.logical_and(P, Q)
        maskXY0 = cp.logical_xor(P, P_and_Q)
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
        return self.anyPerpendicular().normalized()

    """ TODO: Make Function """

    def isInXYPlane(self, atZ, epsilon=0.001):
        pass

    """ TODO: Make Function """

    def isInYZPlane(self, atX, epsilon=0.001):
        pass

    """ TODO: Make Function """

    def isInZXPlane(self, atY, epsilon=0.001):
        pass

    """ TODO: Make Function """

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001):
        pass

    def norm(self):
        return CupyScalars(cp.linalg.norm(self.v, axis=1))

    def abs(self):
        return CupyVectors(cp.abs(self.v))

    def normalize(self):
        """MUST verify that norm is 0."""
        norm = self.norm().v
        normNot0 = cp.where(norm != 0, norm, 1)
        normalizedVectors = self.v / normNot0[:, None]
        self.v = normalizedVectors
        return self

    def normalized(self):
        """Watch out, does this modifies the self also?, yes which is why I deepcopy"""
        v = copy.deepcopy(self)
        return v.normalize()

    """ TODO: Test Function """

    def cross(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.cross(self.v, other.v))
        else:
            return CupyVectors(cp.cross(self.v, other))

    def dot(self, other):
        # element-wise dot product(fake cp.dot)
        # https://stackoverflow.com/questions/41443444/numpy-element-wise-dot-product
        if isinstance(other, CupyVectors):
            return CupyScalars(cp.einsum('ij,ij->i', self.v, other.v))
        else:
            return CupyScalars(cp.einsum('ij,ij->i', self.v, other))

    """ TODO: Test Function """

    def normalizedCrossProduct(self, other):
        '''TODO:  Is this OK'''
        productNorm = self.norm() * other.normSquared()
        return self.cross(other) * productNorm * cp.exp(-0.5)

    """ TODO: Test Function """

    def normalizedDotProduct(self, other):
        '''TODO:  find way to calculate the zeors'''
        productNorm = self.norm() * other.normSquared()
        return self.dot(other) * productNorm * cp.exp(-0.5)

    """ TODO: Test Function """

    def angleWith(self, v, axis):
        """ will v and axis be Vectors Array too or single vectors??"""
        sinPhi = self.normalizedCrossProduct(v)
        sinPhiAbs = sinPhi.abs()
        phi = cp.asin(sinPhiAbs.v)  # what happens here?
        piMinusPhi = cp.pi - phi  # and here ?
        dotV = self.dot(v)
        dotAxis = sinPhi.dot(axis)

        phi = cp.where(dotV.v <= 0, piMinusPhi, phi)
        minusPhi = -phi
        phi = cp.where(dotAxis.v <= 0, minusPhi, phi)

        # print(phi)
        return phi  # What's supposed to be the return type?

    """ TODO: Test Function """

    def planeOfIncidence(self, normal):

        dotNormal = self.dot(normal)
        normal = cp.where(dotNormal.v < 0, -normal, normal)

        planeOfIncidenceNormal = self.cross(normal)
        planeNorm = planeOfIncidenceNormal.norm()

        anyPerp = self.anyPerpendicular()
        planeNormalized = planeOfIncidenceNormal.normalized()

        output = cp.where(planeNorm.v < 1e-7, anyPerp.v, planeNormalized.v)
        return CupyVectors(output)

    """ TODO: Test Function """

    def angleOfIncidence(self, normal):
        dotNormal = self.dot(normal)
        normal = CupyVectors(cp.where(dotNormal.v < 0, -normal, normal))

        planeNormal = self.planeOfIncidence(normal)
        return self.angleWith(normal, axis=planeNormal), planeNormal, normal

    """ TODO: Test Function """

    def rotateAround(self, u, theta):
        u.normalize()
        # print(theta.v)
        if type(theta) == np.ndarray:
            theta = cp.array(theta)
        elif type(theta) == CupyScalars:
            theta = theta.v

        cost = (cp.cos(theta))[0]
        sint = (cp.sin(theta))[0]
        one_cost = (1 - cost)

        ux = u.v[:, 0]
        uy = u.v[:, 1]
        uz = u.v[:, 2]

        X = self.v[:, 0]
        Y = self.v[:, 1]
        Z = self.v[:, 2]

        x = (cost + ux * ux * one_cost) * X + (ux * uy * one_cost - uz * sint) * Y + (
                    ux * uz * one_cost + uy * sint) * Z
        y = (uy * ux * one_cost + uz * sint) * X + (cost + uy * uy * one_cost) * Y + (
                    uy * uz * one_cost - ux * sint) * Z
        z = (uz * ux * one_cost - uy * sint) * X + (uz * uy * one_cost + ux * sint) * Y + (
                    cost + uz * uz * one_cost) * Z

        self.v = cp.stack((x, y, z), axis=-1)
        # print(self.v)

        return self


Vectors = NativeVectors
