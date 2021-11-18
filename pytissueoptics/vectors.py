import pytissueoptics.vector as vec
import pytissueoptics.scalars as sc
import numpy as np
import copy
import pyopencl as pycl
from collections.abc import Iterable
try:
    import cupy as cp
except:
    cp = np

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
    """ This is the reference implementation of NativeVectors. Other classes will
    be created such as GPUNativeVectors, NumpyNativeVectors, CuPyNativeVectors, and others to refine the
    implementation for speed. 
    """

    def __init__(self, vectors=None, N=None):
        self.v = []
        if vectors is not None:
            for v in vectors:
                if v is not None:
                    self.v.append(vec.Vector(v))  # always copy
                else:
                    self.v.append(None)
        elif N is not None:
            self.v = [vec.Vector(0, 0, 0)] * N
        self._iteration = 0
        self.selected = sc.NativeScalars([True] * len(self.v))

    def selectAll(self):
        self.selected = sc.NativeScalars([True] * len(self.v))

    def select(self, which):
        self.selected = sc.NativeScalars(which)

    def selectedNativeVectors(self):
        return NativeVectors([v1 if e else None for (v1, e) in list(zip(self.v, self.selected))])

    @property
    def count(self):
        return len(self.v)

    @classmethod
    def random(cls, N):
        vectors = []
        for i in range(N):
            x = np.random.random() * 2 - 1
            y = np.random.random() * 2 - 1
            z = np.random.random() * 2 - 1
            vectors.append(vec.Vector(x, y, z))
        return NativeVectors(vectors)

    @classmethod
    def randomUnitary(cls, N):
        vectors = []
        for i in range(N):
            x = np.random.random() * 2 - 1
            y = np.random.random() * 2 - 1
            z = np.random.random() * 2 - 1
            vectors.append(vec.Vector(x, y, z).normalized())
        return NativeVectors(vectors)

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
        return NativeVectors(
            [v1 if not e else v1 + v2 * s for (v1, v2, s, e) in list(zip(self.v, rhs, scale, self.selected))])

    @property
    def isUnitary(self) -> [bool]:
        return [v.isUnitary if e else False for v, e in list(zip(self.v, self.selected))]

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
        return NativeVectors([v1 * s if e else v1 for (v1, s, e) in list(zip(self.v, scale, self.selected))])

    def __rmul__(self, scale):
        return NativeVectors([v1 * s if e else v1 for (v1, s, e) in list(zip(self.v, scale, self.selected))])

    def __truediv__(self, scale):
        return NativeVectors([v1 / s if e else v1 for (v1, s, e) in list(zip(self.v, scale, self.selected))])

    def __add__(self, rhs):
        return NativeVectors([v1 + v2 if e else v1 for (v1, v2, e) in list(zip(self.v, rhs.v, self.selected))])

    def __neg__(self):
        return NativeVectors([-v1 if e else v1 for v1, e in list(zip(self.v, self.selected))])

    def __sub__(self, rhs):
        return NativeVectors([v1 - v2 if e else v1 for (v1, v2, e) in list(zip(self.v, rhs.v, self.selected))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue):
        self.v[index] = newvalue

    def __eq__(self, rhs):
        each = [v1 == v2 if e else False for (v1, v2, e) in list(zip(self.v, rhs, self.selected))]
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
        return sc.NativeScalars([False if not e or v1 is None or v2 is None else v1.isEqualTo(v2) for (v1, v2, e) in
                                 list(zip(self.v, rhs, self.selected))])

    def isAlmostEqualTo(self, rhs, epsilon):
        return sc.NativeScalars(
            [False if not e or v1 is None or v2 is None else v1.isAlmostEqualTo(v2, epsilon) for (v1, v2, e) in
             list(zip(self.v, rhs, self.selected))])

    def isParallelTo(self, rhs, epsilon=1e-7):
        return sc.NativeScalars([False if not e or v1 is None else v1.isParallelTo(v2) for (v1, v2, e) in
                                 list(zip(self.v, rhs, self.selected))])

    def isPerpendicularTo(self, rhs, epsilon=1e-7):
        return sc.NativeScalars([False if not e or v1 is None else v1.isPerpendicularTo(v2) for (v1, v2, e) in
                                 list(zip(self.v, rhs, self.selected))])

    def anyPerpendicular(self):
        return NativeVectors(
            [vec.oHat if not e or v1 is None else v1.anyPerpendicular() for v1, e in list(zip(self.v, self.selected))])

    def anyUnitaryPerpendicular(self):
        return NativeVectors([vec.oHat if not e or v1 is None else v1.anyUnitaryPerpendicular() for v1, e in
                              list(zip(self.v, self.selected))])

    def isInXYPlane(self, atZ, epsilon=0.001):
        return sc.NativeScalars(
            [v1.isInXYPlane(atZ=atZ, epsilon=epsilon) if e else False for v1, e in list(zip(self.v, self.selected))])

    def isInYZPlane(self, atX, epsilon=0.001):
        return sc.NativeScalars(
            [v1.isInYZPlane(atX=atX, epsilon=epsilon) if e else False for v1, e in list(zip(self.v, self.selected))])

    def isInZXPlane(self, atY, epsilon=0.001):
        return sc.NativeScalars(
            [v1.isInZXPlane(atY=atY, epsilon=epsilon) if e else False for v1, e in list(zip(self.v, self.selected))])

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001) -> bool:
        return sc.NativeScalars(
            [v1.isInPlane(origin, normal, epsilon) if e else False for v1, e in list(zip(self.v, self.selected))])

    def norm(self):
        return sc.NativeScalars([v1.normSquared() if e else 0 for v1, e in list(zip(self.v, self.selected))])

    def abs(self):
        return sc.NativeScalars([v1.abs() if e else 0 for v1, e in list(zip(self.v, self.selected))])

    def normalize(self):
        [v1.normalize() if e else v1 for v1, e in list(zip(self.v, self.selected))]
        return self

    def normalized(self):
        return NativeVectors([v1.normalized() if e else v1 for v1, e in list(zip(self.v, self.selected))])

    def cross(self, rhs):
        return NativeVectors([v1.cross(v2) if e else v1 for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def dot(self, rhs):
        return sc.NativeScalars([v1.dot(v2) if e else 0 for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def normalizedCrossProduct(self, rhs):
        return NativeVectors([v1.normalizedCrossProduct(v2) if e else v1
                              for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def normalizedDotProduct(self, rhs):
        return sc.NativeScalars(
            [v1.normalizedDotProduct(v2) if e else 0 for (v1, v2, e) in list(zip(self.v, rhs, self.selected))])

    def angleWith(self, v, axis):
        return sc.NativeScalars(
            [v1.angleWith(v=v2, axis=v3) if e else 0 for (v1, v2, v3, e) in list(zip(self.v, v, axis, self.selected))])

    def planeOfIncidence(self, normal):
        return NativeVectors(
            [v1.planeOfIncidence(normal=v2) if e else vec.oHat for (v1, v2, e) in list(zip(self.v, normal, self.selected))])

    def angleOfIncidence(self, normal):
        dotProduct = self.dot(normal)
        normal.selected = (dotProduct.v < 0)
        correctedNormal = -normal

        planeNormal = self.planeOfIncidence(correctedNormal)
        angles = sc.NativeScalars(self.angleWith(correctedNormal, axis=planeNormal))
        return angles, planeNormal, correctedNormal

    def rotateAround(self, u, theta):
        [v1.rotateAround(v2, t) if e else v1 for (v1, v2, t, e) in list(zip(self.v, u.v, theta, self.selected))]
        return self

    def rotatedAround(self, u, theta):
        v = NativeVectors(self)  # copy
        [v1.rotateAround(v2, t) if e else v1 for (v1, v2, t, e) in list(zip(v.v, u.v, theta, self.selected))]
        return v


class NumpyVectors:
    """ This is the Reference Vectors Class for numpy-like calculations.
    This architecture will be used by cupy since it is a drop-in replacement
    It is the go-to fallback since numpy is available for every python user
    and since it doesn't require a GPU.
    """

    def __init__(self, vectors=None, N=None):
        self.v = np.array([[None, None, None]])

        if vectors is not None and N is None:

            if type(vectors) == np.ndarray:
                    self.v = np.atleast_2d(vectors.astype('float64'))

            elif type(vectors) == vec.Vector:
                self.v = np.asarray([[vectors.x, vectors.y, vectors.z]], dtype=np.float64)

            elif type(vectors) == list and isinstance(vectors[0], vec.Vector):
                x = [v.x for v in vectors]
                y = [v.y for v in vectors]
                z = [v.z for v in vectors]
                self.v = np.stack((x, y, z), axis=-1)

            else:
                self.v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))

        elif vectors is not None and N is not None:
            if type(vectors) == vec.Vector:
                self.v = np.asarray([[vectors.x, vectors.y, vectors.z]] * N, dtype=np.float64)

            elif type(vectors) == list and type(vectors[0]) != list:
                self.v = np.asarray([vectors] * N, dtype=np.float64)

            elif type(vectors) == np.ndarray:
                self.v = np.tile(vectors, (N, 1))

            else:
                self.v = np.asarray(vectors * N, dtype=np.float64)

        elif vectors is None and N is not None:
            self.v = np.zeros((N, 3), dtype=np.float64)

        self._iteration = 0

    def __len__(self):
        return self.v.shape[0]

    def __mul__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.multiply(self.v, other.v))
        elif isinstance(other, sc.NumpyScalars):
            return ArithmeticError
        elif type(other) in (float, int):
            return NumpyVectors(np.multiply(self.v, other))

    def __rmul__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.multiply(other.v, self.v))
        elif isinstance(other, sc.NumpyScalars):
            return NumpyVectors(np.multiply(self.v, other.v[:, None]))
        elif type(other) in (float, int):
            return NumpyVectors(np.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, NumpyVectors):
            return NumpyVectors(np.true_divide(self.v, other.v))
        elif isinstance(other, sc.NumpyScalars):
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
        elif isinstance(other, sc.NumpyScalars):
            return NumpyVectors(np.subtract(self.v, other.v))
        elif isinstance(other, vec.Vector):
            return NumpyVectors(np.subtract(self.v, list(other)))
        else:
            return NumpyVectors(np.subtract(self.v, other))

    def __neg__(self):
        return NumpyVectors(np.negative(self.v))

    def __eq__(self, other):
        if isinstance(other, (NumpyVectors, sc.NumpyScalars)):
            return sc.NumpyScalars(np.where(self.v == other.v, 1, 0))
        else:
            return sc.NumpyScalars(np.where(self.v == other, 1, 0))

    def __ne__(self, other):
        if isinstance(other, (NumpyVectors, sc.NumpyScalars)):
            return sc.NumpyScalars(np.where(self.v != other.v, 1, 0))
        else:
            return sc.NumpyScalars(np.where(self.v != other, 1, 0))

    def __lt__(self, other):
        if isinstance(other, (NumpyVectors, sc.NumpyScalars)):
            return sc.NumpyScalars(np.where(self.v < other.v, 1, 0))
        else:
            return sc.NumpyScalars(np.where(self.v < other, 1, 0))

    def __gt__(self, other):
        if isinstance(other, (NumpyVectors, sc.NumpyScalars)):
            return sc.NumpyScalars(np.where(self.v > other.v, 1, 0))
        else:
            return sc.NumpyScalars(np.where(self.v > other, 1, 0))

    def __le__(self, other):
        if isinstance(other, (NumpyVectors, sc.NumpyScalars)):
            return sc.NumpyScalars(np.where(self.v <= other.v, 1, 0))
        else:
            return sc.NumpyScalars(np.where(self.v <= other, 1, 0))

    def __ge__(self, other):
        if isinstance(other, (NumpyVectors, sc.NumpyScalars)):
            return sc.NumpyScalars(np.where(self.v >= other.v, 1, 0))
        else:
            return sc.NumpyScalars(np.where(self.v >= other, 1, 0))

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
        if self.v is None:
            raise StopIteration

        if self._iteration < len(self.v):
            result = self.v[self._iteration]
            self._iteration += 1
            return result

        else:
            raise StopIteration

    def append(self, value):
        #  FIXME: Problem with the type (Doesnt get recognized by numpy should force to float64)
        if self.isEmpty:
            self.v = Vectors(value).v
        else:
            refactoredValue = Vectors(value).v
            self.v = np.append(self.v, refactoredValue, axis=0)

    @property
    def isEmpty(self):
        if len(self) == 1:
            if np.all(self.v[0] == [None, None, None]):
                return True
            else:
                return False
        else:
            return False

    @property
    def x(self):
        x = self.v[:, 0]
        return sc.NumpyScalars(x)

    @property
    def y(self):
        y = self.v[:, 1]
        return sc.NumpyScalars(y)

    @property
    def z(self):
        z = self.v[:, 2]
        return sc.NumpyScalars(z)

    @property
    def isUnitary(self):
        return sc.NumpyScalars(np.less(np.abs(np.linalg.norm(self.v, axis=1)) - 1, 1e-9))

    @property
    def isNull(self):
        return sc.NumpyScalars(np.less(np.linalg.norm(self.v, axis=1), 1e-9))

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
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return NumpyVectors(np.stack((x, y, z), axis=-1))

    def isEqualTo(self, other):
        if isinstance(other, NumpyVectors):
            return sc.NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other.v)), 1e-9))
        else:
            return sc.NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other)), 1e-9))

    def isAlmostEqualTo(self, other, epsilon):
        if isinstance(other, NumpyVectors):
            return sc.NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other.v)), epsilon))
        else:
            return sc.NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other)), epsilon))

    def isParallelTo(self, other, epsilon=1e-9):
        r = self.normalizedCrossProduct(other).norm().v
        a = np.less_equal(r, epsilon)
        r = np.where(self.isNull | other.isNull, False, a)
        return sc.NumpyScalars(r)

    def isPerpendicularTo(self, other, epsilon=1e-9):
        r = np.abs(self.normalizedDotProduct(other).v)
        a = np.less_equal(r, epsilon)
        r = np.where(self.isNull | other.isNull, False, a)
        return r

    def anyPerpendicular(self):
        ux = self.v[:, 0]
        uy = self.v[:, 1]
        uz = self.v[:, 2]

        a = np.stack((uy, -ux, np.zeros(len(ux))), axis=-1)
        b = np.stack((np.zeros(len(ux)), -uz, uy), axis=-1)
        c = np.where(uz[:, None] < ux[:, None], a, b)
        r = np.where(np.linalg.norm(c, axis=1)[:, None] != 0, c, None)

        return NumpyVectors(r)

    def anyUnitaryPerpendicular(self):
        return self.anyPerpendicular().normalized()

    def isInXYPlane(self, atZ, epsilon=0.001):
        uz = self.v[:, 2]
        r = np.where(np.abs(uz - atZ) < epsilon, True, False)
        return r

    def isInYZPlane(self, atX, epsilon=0.001):
        ux = self.v[:, 0]
        r = np.where(np.abs(ux - atX) < epsilon, True, False)
        return r

    def isInZXPlane(self, atY, epsilon=0.001):
        uy = self.v[:, 1]
        r = np.where(np.abs(uy - atY) < epsilon, True, False)
        return r

    def isInPlane(self, origin: 'NumpyVectors', normal: 'NumpyVectors', epsilon=0.001):
        local = self - origin
        r = np.where(np.abs(local.normalizedDotProduct(normal)) < epsilon, True, False)
        return r

    def addScaled(self, other, scale):
        ux = other.v[:, 0] * scale
        uy = other.v[:, 1] * scale
        uz = other.v[:, 2] * scale

        X = self.v[:, 0]
        Y = self.v[:, 1]
        Z = self.v[:, 2]

        x = X + ux
        y = Y + uy
        z = Z + uz

        self.v = np.stack((x, y, z), axis=-1)

        return self

    def norm(self):
        return sc.NumpyScalars(np.linalg.norm(self.v, axis=1))

    def normSquared(self):
        return sc.NumpyScalars(self.abs)

    def abs(self):
        return NumpyVectors(np.abs(self.v))

    def normalize(self):
        """MUST verify that norm is 0."""
        norm = self.norm().v
        if not np.all(norm):
            raise ValueError("Normalizing the null vector is impossible.")

        normalizedVectors = self.v / norm[:, None]
        self.v = normalizedVectors
        return self

    def normalized(self):
        # Watch out, does this modifies the self also?, yes which is why I deepcopy()
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
            return sc.NumpyScalars(np.einsum('ij,ij->i', self.v, other.v))
        else:
            return sc.NumpyScalars(np.einsum('ij,ij->i', self.v, other))

    def normalizedCrossProduct(self, other):
        productNorm = (self.norm() * other.norm()).v
        productNorm = np.where(productNorm != 0, productNorm, 1)
        output = self.cross(other) / productNorm[:, None]
        return output

    def normalizedDotProduct(self, other):
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

        return sc.NumpyScalars(phi)  # What's supposed to be the return type?

    def planeOfIncidence(self, normal):
        normVector = self.norm().v
        normPlane = normal.norm().v
        if not (np.all(normVector) and np.all(normPlane)):
            raise ValueError("The direction of incidence or the normal cannot be null")

        dotNormal = self.dot(normal)
        normal = np.where(dotNormal.v[:, None] < 0, -normal.v, normal.v)
        planeOfIncidenceNormal = self.cross(normal)
        planeNorm = planeOfIncidenceNormal.norm()
        anyPerp = self.anyPerpendicular()
        output = np.where(planeNorm.v[:, None] < 1e-3, anyPerp.v, planeOfIncidenceNormal.v)

        return NumpyVectors(output).normalized()

    def angleOfIncidence(self, normal):
        dotNormal = self.dot(normal)
        normal = NumpyVectors(np.where(dotNormal.v[:, None] < 0, -normal.v, normal.v))
        planeNormal = self.planeOfIncidence(normal)

        return self.angleWith(normal, axis=planeNormal), planeNormal, normal

    def rotateAround(self, u, theta):
        u.normalize()
        # print(theta.v)
        cost = (np.cos(theta.v))
        sint = (np.sin(theta.v))
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

        self.v = np.stack((x, y, z), axis=-1)

        return self


class CupyVectors:
    """ This is the Reference Cupy Vectors Class for cupy-like calculations.
    This architecture will be used to do GPU calculations on hosts
    that have CUDA libraries and cupy. This requires a NVIDIA GPU. Will fallback
    to OpenCLVectors or CupyVectors depending on requirements met.
    """

    def __init__(self, vectors=None, N=None):
        if vectors is not None and N is None:
            if type(vectors) == np.ndarray:
                self.v = cp.asarray(vectors, dtype=cp.float64)

            elif type(vectors) == cp.ndarray:
                self.v = vectors.astype('float64')

            elif type(vectors) == vec.Vector:
                self.v = cp.asarray([[vectors.x, vectors.y, vectors.z]], dtype=cp.float64)

            elif type(vectors) == list and type(vectors[0]) == vec.Vector:
                x = cp.asarray([v.x for v in vectors])
                y = cp.asarray([v.y for v in vectors])
                z = cp.asarray([v.z for v in vectors])
                self.v = cp.stack((x, y, z), axis=-1)

            else:
                self.v = cp.asarray(vectors, dtype=cp.float64)

        elif vectors is not None and N is not None:
            if type(vectors) == vec.Vector:
                self.v = cp.asarray([[vectors.x, vectors.y, vectors.z]] * N, dtype=cp.float64)

            elif type(vectors) == list and type(vectors[0]) != list:
                self.v = cp.asarray([vectors] * N, dtype=cp.float64)

            elif type(vectors) == np.ndarray:
                self.v = cp.asarray(np.tile(vectors, (N, 1)))

            else:
                self.v = cp.asarray(vectors * N, dtype=cp.float64)

        elif vectors is None and N is not None:
            self.v = cp.zeros((N, 3), dtype=cp.float64)

        self._iteration = 0

    def __len__(self):
        return self.v.shape[0]

    def __mul__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.multiply(self.v, other.v))
        elif isinstance(other, sc.CupyScalars):
            return CupyVectors(cp.multiply(self.v, other.v[:, None]))
        # elif isinstance(other, cp.ndarray):
        #     if len(other.shape) == 1:
        #         return CupyVectors(self.v * other[:, None])
        else:
            return CupyVectors(cp.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.true_divide(self.v, other.v))
        elif isinstance(other, sc.CupyScalars):
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
    def x(self):
        x = self.v[:, 0]
        return sc.CupyScalars(x)

    @property
    def y(self):
        y = self.v[:, 1]
        return sc.CupyScalars(y)

    @property
    def z(self):
        z = self.v[:, 2]
        return sc.CupyScalars(z)

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
        theta = (cp.np.random.rand(N) * 2 * cp.pi)
        phi = (cp.np.random.rand(N) * cp.pi)
        x = r * cp.sin(phi) * cp.cos(theta)
        y = r * cp.sin(phi) * cp.sin(theta)
        z = r * cp.cos(phi)
        return CupyVectors(cp.stack((x, y, z), axis=-1))

    @classmethod
    def randomUniformUnitary(cls, N):
        theta = cp.np.random.rand(N) * 2 * cp.pi
        phi = cp.np.random.rand(N) * cp.pi
        x = cp.sin(phi) * cp.cos(theta)
        y = cp.sin(phi) * cp.sin(theta)
        z = cp.cos(phi)
        output = CupyVectors(cp.stack((x, y, z), axis=-1))
        print(output)
        return output

    def isEqualTo(self, other):
        if isinstance(other, CupyVectors):
            return sc.CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other.v)), 1e-9))
        else:
            return sc.CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other)), 1e-9))

    def isAlmostEqualTo(self, other, epsilon):
        if isinstance(other, CupyVectors):
            return sc.CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other.v)), epsilon))
        else:
            return sc.CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other)), epsilon))

    def isParallelTo(self, other, epsilon=1e-9):
        r = self.normalizedCrossProduct(other).norm().v
        a = cp.less_equal(r, epsilon)
        r = cp.where(self.isNull | other.isNull, False, a)
        return r

    def isPerpendicularTo(self, other, epsilon=1e-9):
        r = cp.abs(self.normalizedDotProduct(other).v)
        a = cp.less_equal(r, epsilon)
        r = cp.where(self.isNull | other.isNull, False, a)
        return r

    def anyPerpendicular(self):
        ux = self.v[:, 0]
        uy = self.v[:, 1]
        uz = self.v[:, 2]

        a = cp.stack((uy, -ux, cp.zeros(len(ux))), axis=-1)
        b = cp.stack((cp.zeros(len(ux)), -uz, uy), axis=-1)
        c = cp.where(uz[:, None] < ux[:, None], a, b)
        # not verifying the null vector as it should never happen and cupy doesn't like this syntax

        return CupyVectors(c)

    def anyUnitaryPerpendicular(self):
        return self.anyPerpendicular().normalized()

    def isInXYPlane(self, atZ, epsilon=0.001):
        uz = self.v[:, 2]
        r = cp.where(cp.abs(uz - atZ) < epsilon, True, False)
        return r

    def isInYZPlane(self, atX, epsilon=0.001):
        ux = self.v[:, 0]
        r = cp.where(cp.abs(ux - atX) < epsilon, True, False)
        return r

    def isInZXPlane(self, atY, epsilon=0.001):
        uy = self.v[:, 1]
        r = cp.where(cp.abs(uy - atY) < epsilon, True, False)
        return r

    def isInPlane(self, origin: 'NumpyVectors', normal: 'NumpyVectors', epsilon=0.001):
        local = self - origin
        r = cp.where(cp.abs(local.normalizedDotProduct(normal)) < epsilon, True, False)
        return r

    def norm(self):
        return sc.CupyScalars(cp.linalg.norm(self.v, axis=1))

    def normSquared(self):
        return sc.CupyScalars(self.abs)

    def abs(self):
        return CupyVectors(cp.abs(self.v))

    def addScaled(self, other, scale):
        ux = other.v[:, 0] * scale.v
        uy = other.v[:, 1] * scale.v
        uz = other.v[:, 2] * scale.v

        X = self.v[:, 0]
        Y = self.v[:, 1]
        Z = self.v[:, 2]

        x = X + ux
        y = Y + uy
        z = Z + uz

        self.v = cp.stack((x, y, z), axis=-1)

        return self

    def normalize(self):
        """MUST verify that norm is 0."""
        norm = self.norm().v
        if not cp.all(norm):
            raise ValueError("Normalizing the null vector is impossible.")

        normalizedVectors = self.v / norm[:, None]
        self.v = normalizedVectors
        return self

    def normalized(self):
        # Watch out, does this modifies the self also?, yes which is why I deepcopy()
        v = copy.deepcopy(self)
        return v.normalize()

    def cross(self, other):
        if isinstance(other, CupyVectors):
            return CupyVectors(cp.cross(self.v, other.v))
        else:
            return CupyVectors(cp.cross(self.v, other))

    def dot(self, other):
        # element-wise dot product(fake cp.dot)
        # https://stackoverflow.com/questions/41443444/numpy-element-wise-dot-product
        if isinstance(other, CupyVectors):
            return sc.CupyScalars(cp.einsum('ij,ij->i', self.v, other.v))
        else:
            return sc.CupyScalars(cp.einsum('ij,ij->i', self.v, other))

    def normalizedCrossProduct(self, other):
        productNorm = (self.norm() * other.norm()).v
        productNorm = cp.where(productNorm != 0, productNorm, 1)
        output = self.cross(other) / productNorm[:, None]
        return output

    def normalizedDotProduct(self, other):
        invAbs = (self.norm() * other.norm()).v
        invAbs = cp.where(invAbs != 0, invAbs, 1)
        dot = self.dot(other)
        output = dot / invAbs
        return output

    def angleWith(self, v, axis):
        """ will v and axis be Vectors Array too or single vectors??"""
        sicphi = self.normalizedCrossProduct(v)
        sicphiAbs = sicphi.norm()
        phi = cp.arcsin(sicphiAbs.v)
        piMinusPhi = cp.pi - phi
        dotV = self.dot(v)
        dotAxis = sicphi.dot(axis)

        phi = cp.where(dotV.v <= 0, piMinusPhi, phi)
        minusPhi = -phi
        phi = cp.where(dotAxis.v <= 0, minusPhi, phi)

        return sc.CupyScalars(phi)  # What's supposed to be the return type?

    def planeOfIncidence(self, normal):
        normVector = self.norm().v
        normPlane = normal.norm().v
        if not (cp.all(normVector) and cp.all(normPlane)):
            raise ValueError("The direction of incidence or the normal cannot be null")

        dotNormal = self.dot(normal)
        normal = cp.where(dotNormal.v[:, None] < 0, -normal.v, normal.v)
        planeOfIncidenceNormal = self.cross(normal)
        planeNorm = planeOfIncidenceNormal.norm()
        anyPerp = self.anyPerpendicular()
        output = cp.where(planeNorm.v[:, None] < 1e-3, anyPerp.v, planeOfIncidenceNormal.v)

        return CupyVectors(output).normalized()

    def angleOfIncidence(self, normal):
        dotNormal = self.dot(normal)
        normal = CupyVectors(cp.where(dotNormal.v[:, None] < 0, -normal.v, normal.v))
        planeNormal = self.planeOfIncidence(normal)

        return self.angleWith(normal, axis=planeNormal), planeNormal, normal

    """ TODO: Test Function """

    def rotateAround(self, u, theta):
        u.normalize()
        # print(theta.v)
        cost = (cp.cos(theta.v))
        sint = (cp.sin(theta.v))
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

        return self


class GPUManager:
    def __init__(self):
        self.platforms = [platform for platform in pycl.get_platforms()]
        self.devices = self.platforms[0].get_devices()
        self.context = pycl.Context(devices=self.devices)
        program_source = """
              kernel void sum(global float *a, 
                              global float *b,
                              global float *c){
                int gid = get_global_id(0);
                c[gid] = a[gid] + b[gid];
              }
                kernel void multiply(global float *a, 
                              global float *b,
                              global float *c){
                int gid = get_global_id(0);
                c[gid] = a[gid] * b[gid];
              }
            """
        program_source = pycl.Program(self.context, program_source)
        self.program = program_source.build()

    def createReadWriteMemoryBuffer(self, context, arrays):
        return [(array, pycl.Buffer(context,
                                    flags=pycl.mem_flags.WRITE_ONLY,
                                    size=array.nbytes))
                for array in arrays]

    def copyNewVectorAndLinkPointer(self, newVector):
        pass


class OpenclVectors:
    def __init__(self, vectors=None, N=None):
        if vectors is not None:
            if type(vectors) == np.ndarray:
                self.v = vectors.astype('float32')

            else:
                self.v = np.asarray(vectors, dtype=np.float64)
        elif N is not None:
            self.v = np.zeros((N, 3), dtype=np.float64)

        self._iteration = 0

Vectors = NumpyVectors