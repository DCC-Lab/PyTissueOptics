import numpy as np

try:
    import cupy as cp
except:
    cp = np
    # CupyScalars = NumpyScalars

import arrayImplementation.vectors as vc


class NativeScalars:
    """ An array of scalars that is compatible with operations on Vectors 
    There is a reason for not using numpy.array directly: we want to
    add new functions that will be specific to our problem here,
    and Python does not allow us to extend system classes.

    The class works with float, int and bool.  With boolean values, True is 1.
    """

    def __init__(self, array=None, N=None):
        """

        @rtype: object
        """
        if array is not None:
            self.v = np.array(array)
        elif N is not None:
            self.v = np.array([0] * N)
        else:
            raise ValueError("You must provide an array or N")
        self.selected = [True] * len(self.v)
        self._iteration = 0

    @classmethod
    def random(cls, N):
        return Scalars([np.random.random() for i in range(N)])

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < len(self):
            result = self.v[self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.v)

    def __mul__(self, scale) -> 'Scalars':
        return Scalars(self.v * scale)

    def __rmul__(self, scale) -> 'Scalars':
        return Scalars(self.v * scale)

    def __truediv__(self, scale) -> 'Scalars':
        return Scalars(self.v / scale)

    def __add__(self, rhs) -> 'Scalars':
        return Scalars([v1 + v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __neg__(self) -> 'Scalars':
        return Scalars([-v1 for v1 in self.v])

    def __sub__(self, rhs) -> 'Scalars':
        return Scalars([v1 - v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue):
        self.v[index] = newvalue

    def __eq__(self, rhs) -> bool:
        if isinstance(rhs, Scalars):
            each = [v1 == v2 for (v1, v2) in list(zip(self.v, rhs.v))]
        else:
            each = [v1 == v2 for (v1, v2) in list(zip(self.v, rhs))]
        return np.array(each).all()

    def logicalNot(self) -> 'Scalars':
        return Scalars([not bool(v1) for v1 in self.v])

    def logicalAnd(self, rhs) -> 'Scalars':
        return Scalars([bool(v1) and bool(v2) for v1, v2 in list(zip(self.v, rhs))])

    def logicalOr(self, rhs) -> 'Scalars':
        return Scalars([bool(v1) or bool(v2) for v1, v2 in list(zip(self.v, rhs))])

    def all(self) -> bool:
        return self.v.all()

    def any(self) -> bool:
        return self.v.any()

    def none(self) -> bool:
        return not self.v.any()


class NumpyScalars:
    def __init__(self, array=None, N=None):
        if array is not None:
            if type(array) == np.ndarray:
                self.v = array.astype('float64')
            elif type(array) == cp.ndarray:
                self.v = array.astype(np.float64)
            else:
                self.v = np.asarray(array, dtype=np.float64)
        elif N is not None:
            self.v = np.zeros((1, N), dtype=np.float64)

        self._iteration = 0

    def __len__(self):
        return self.v.shape[1]

    def __add__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.add(self.v, other.v))
        else:
            return NumpyScalars(np.add(self.v, other))

    def __sub__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.subtract(self.v, other.v))
        else:
            return NumpyScalars(np.subtract(self.v, other))

    def __mul__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.multiply(self.v, other.v))
        elif isinstance(other, vc.NumpyVectors):
            return NumpyScalars(np.multiply(self.v[:, None], other.v))
        else:
            return NumpyScalars(np.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.true_divide(self.v, other.v))
        elif isinstance(other, vc.NumpyVectors):
            return NumpyScalars(np.multiply(self.v[:, None], other.v))
        else:
            return NumpyScalars(np.true_divide(self.v, other))

    def __neg__(self):
        return NumpyScalars(np.negative(self.v))

    def __invert__(self):
        return NumpyScalars(np.logical_not(self.v))

    def __getitem__(self, item):
        return self.v[item]

    def __setitem__(self, key, value: np.float32):
        self.v[key] = value

    def __eq__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v == other.v, 1, 0))
        else:
            return NumpyScalars(np.where(self.v == other, 1, 0))

    def __ne__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v != other.v, 1, 0))
        else:
            return NumpyScalars(np.where(self.v != other, 1, 0))

    def __lt__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v < other.v, 1, 0))
        else:
            return NumpyScalars(np.where(self.v < other, 1, 0))

    def __gt__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v > other.v, 1, 0))
        else:
            return NumpyScalars(np.where(self.v > other, 1, 0))

    def __le__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v <= other.v, 1, 0))
        else:
            return NumpyScalars(np.where(self.v <= other, 1, 0))

    def __ge__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v >= other.v, 1, 0))
        else:
            return NumpyScalars(np.where(self.v >= other, 1, 0))

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < len(self):
            result = self.v[:, self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    def __contains__(self, item):
        if item in self.v:
            return True
        else:
            return False

    def logical_and(self, other):
        return NumpyScalars(np.logical_and(self.v, other.v))

    @classmethod
    def setAll(cls, value, N):
        return NumpyScalars(np.full((N), value))

    @classmethod
    def random(cls, N: int):
        """Random number between [0, 1]"""
        return NumpyScalars(np.random.rand(N))

    @classmethod
    def random2(cls, N: int):
        """Random number between [-1, 1]"""
        return NumpyScalars((np.random.rand(N) * 2) - 1)

    def isEqualTo(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other.v)), 1e-9))
        else:
            return NumpyScalars(np.less_equal(np.abs(np.subtract(self.v, other)), 1e-9))

    def condition(self, condition, value, a="same", b="same"):
        if condition == ">=":
            np.where(self.v >= value,)
        elif condition == "==":
            pass
        elif condition == "<=":
            pass
        elif condition == "!=":
            pass


class CupyScalars:
    def __init__(self, array=None, N=None):
        if array is not None:
            if type(array) == cp.ndarray:
                self.v = array.astype('float64')
            elif type(array) == cp.ndarray:
                self.v = array.astype(cp.float64)
            else:
                self.v = cp.asarray(array, dtype=cp.float64)
        elif N is not None:
            self.v = cp.zeros((1, N), dtype=cp.float64)

        self._iteration = 0

    def __len__(self):
        return self.v.shape[1]

    def __add__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.add(self.v, other.v))
        else:
            return CupyScalars(cp.add(self.v, other))

    def __sub__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.subtract(self.v, other.v))
        else:
            return CupyScalars(cp.subtract(self.v, other))

    def __mul__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.multiply(self.v, other.v))
        elif isinstance(other, vc.CupyVectors):
            return CupyScalars(cp.multiply(self.v[:, None], other.v))
        else:
            return CupyScalars(cp.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.true_divide(self.v, other.v))
        elif isinstance(other, vc.CupyVectors):
            return CupyScalars(cp.multiply(self.v[:, None], other.v))
        else:
            return CupyScalars(cp.true_divide(self.v, other))

    def __neg__(self):
        return CupyScalars(cp.negative(self.v))

    def __getitem__(self, item):
        return self.v[item]

    def __setitem__(self, key, value: cp.float32):
        self.v[key] = value

    def __eq__(self, other):
        if isinstance(other, CupyScalars):
            return cp.equal(self.v, other.v)
        else:
            return cp.equal(self.v, other)

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < len(self):
            result = self.v[:, self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    def __contains__(self, item):
        if item in self.v:
            return True
        else:
            return False

    @classmethod
    def setAll(cls, value, N):
        return CupyScalars(cp.full((N), value))

    @classmethod
    def random(cls, N: int):
        """Random number between [0, 1]"""
        return CupyScalars(cp.random.rand(N))

    @classmethod
    def random2(cls, N: int):
        """Random number between [-1, 1]"""
        return CupyScalars((cp.random.rand(N) * 2) - 1)

    def isEqualTo(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other.v)), 1e-9))
        else:
            return CupyScalars(cp.less_equal(cp.abs(cp.subtract(self.v, other)), 1e-9))


Scalars = NumpyScalars
