import pytissueoptics.vectors as vec
import numpy as np
try:
    import cupy as cp
except:
    cp = np


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
        self.v = []
        if array is not None:
            self.v = np.atleast_1d(np.asarray(array, dtype=np.float32)).tolist()
        elif N is not None:
            self.v = np.array([0] * N).tolist()

        self._iteration = 0

    @property
    def selected(self):
        if len(self) == 0:
            return None
        else:
            selected = [True] * len(self.v)
            return selected

    @classmethod
    def random(cls, N):
        return NativeScalars([np.random.random() for i in range(N)])

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
        return NativeScalars(self.v * scale)

    def __rmul__(self, scale) -> 'Scalars':
        return NativeScalars(self.v * scale)

    def __truediv__(self, scale) -> 'Scalars':
        return NativeScalars(self.v / scale)

    def __add__(self, rhs) -> 'Scalars':
        return NativeScalars([v1 + v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __neg__(self) -> 'Scalars':
        return NativeScalars([-v1 for v1 in self.v])

    def __sub__(self, rhs) -> 'Scalars':
        return NativeScalars([v1 - v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue):
        self.v[index] = newvalue

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

    def append(self, value):
        if value is not None:
            a = NativeScalars(value)
            for element in a:
                self.v.append(element)

    def all(self) -> bool:
        return self.v.all()

    def any(self) -> bool:
        return self.v.any()

    def none(self) -> bool:
        return not self.v.any()

    @property
    def isEmpty(self):
        if len(self) == 0:
            return True
        else:
            return False

    @property
    def isBool(self):
        for v in self.v:
            if v != 0 and v != 1:
                return False
            else:
                continue
        return True

    def logical_and(self, other):
        if isinstance(other, NativeScalars):
            if self.isBool and other.isBool:
                return NativeScalars([bool(v1) and bool(v2) for v1, v2 in list(zip(self.v, other.v))])
        else:
            if self.isBool:
                return NativeScalars(cp.logical_and(self.v, other))

    def logical_or(self, other):
        if isinstance(other, NativeScalars):
            if self.isBool and other.isBool:
                return NativeScalars([bool(v1) or bool(v2) for v1, v2 in list(zip(self.v, other.v))])
        else:
            if self.isBool:
                return NativeScalars([bool(v1) or bool(v2) for v1, v2 in list(zip(self.v, other))])

    def logical_xor(self, other):
        if isinstance(other, NativeScalars):
            if self.isBool and other.isBool:
                return NativeScalars([bool(v1) != bool(v2) for v1, v2 in list(zip(self.v, other.v))])
        else:
            if self.isBool:
                return NativeScalars([bool(v1) != bool(v2) for v1, v2 in list(zip(self.v, other))])

    def conditional_le(self, other, a: list, b: list):
        if isinstance(other, NativeScalars):
            return NativeScalars([a[i] if self.v[i]<=other[i] else b[i] for i in range(len(self.v))])
        elif type(other) in (int, float):
            return NativeScalars([a[i] if self.v[i]<=other else b[i] for i in range(len(self.v))])

    def conditional_lt(self, other, a, b):
        if isinstance(other, NativeScalars):
            return NativeScalars([a[i] if self.v[i] < other[i] else b[i] for i in range(len(self.v))])
        elif type(other) in (int, float):
            return NativeScalars([a[i] if self.v[i] < other else b[i] for i in range(len(self.v))])

    def conditional_gt(self, other, a, b):
        if isinstance(other, NativeScalars):
            return NativeScalars([a[i] if self.v[i] > other[i] else b[i] for i in range(len(self.v))])
        elif type(other) in (int, float):
            return NativeScalars([a[i] if self.v[i] > other else b[i] for i in range(len(self.v))])

    def conditional_ge(self, other, a, b):
        if isinstance(other, NativeScalars):
            return NativeScalars([a[i] if self.v[i] >= other[i] else b[i] for i in range(len(self.v))])
        elif type(other) in (int, float):
            return NativeScalars([a[i] if self.v[i] >= other else b[i] for i in range(len(self.v))])

    def conditional_eq(self, other, a, b):
        if isinstance(other, NativeScalars):
            return NativeScalars([a[i] if self.v[i] == other[i] else b[i] for i in range(len(self.v))])
        elif type(other) in (int, float):
            return NativeScalars([a[i] if self.v[i] == other else b[i] for i in range(len(self.v))])

    def conditional_neq(self, other, a, b):
        if isinstance(other, NativeScalars):
            return NativeScalars([a[i] if self.v[i] != other[i] else b[i] for i in range(len(self.v))])
        elif type(other) in (int, float):
            return NativeScalars([a[i] if self.v[i] != other else b[i] for i in range(len(self.v))])


class NumpyScalars:
    def __init__(self, array=None, N=None, ndtype="float64"):
        self.v = np.array([])
        if array is not None:
            if type(array) == np.ndarray:
                self.v = np.atleast_1d(array.astype(ndtype))
            elif type(array) == cp.ndarray:
                self.v = np.atleast_1d(array.astype(ndtype))
            else:
                self.v = np.atleast_1d(np.asarray(array, dtype=ndtype))
        elif N is not None:
            self.v = np.zeros((N), dtype=ndtype)

        self._iteration = 0

    @property
    def isEmpty(self):
        if len(self) == 0:
            return True
        else:
            return False

    def append(self, value):
        if self.isEmpty:
            self.v = Scalars(value).v
        else:
            refactoredValue = Scalars(value).v
            self.v = np.append(self.v, refactoredValue)

    def __repr__(self):
        return str(self.v)

    def __len__(self):
        return self.v.shape[0]

    def __add__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.add(self.v, other.v))
        elif isinstance(other, vec.NumpyVectors):
            return ArithmeticError
        else:
            return NumpyScalars(np.add(self.v, other))

    def __sub__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.subtract(self.v, other.v))
        elif isinstance(other, vec.NumpyVectors):
            return ArithmeticError
        elif type(other) in (float, int):
            return NumpyScalars(np.subtract(self.v, other))

    def __mul__(self, other):
        if isinstance(other, NumpyScalars):
            if len(self) == len(other):
                return NumpyScalars(np.multiply(self.v, other.v))
        elif isinstance(other, vec.NumpyVectors):
            if len(self) == len(other):
                return vec.NumpyVectors(np.multiply(self.v[:, None], other.v))
        elif type(other) in (float, int):
            return NumpyScalars(np.multiply(self.v, other))

    def __rmul__(self, other):
        if isinstance(other, NumpyScalars):
            if len(self) == len(other):
                return NumpyScalars(np.multiply(self.v, other.v))
        elif isinstance(other, vec.NumpyVectors):
            return ArithmeticError
        elif type(other) in (float, int):
            return NumpyScalars(np.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.true_divide(self.v, other.v))
        elif isinstance(other, vec.NumpyVectors):
            return ArithmeticError
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
            result = self.v[self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    def __contains__(self, item):
        if item in self.v:
            return True
        else:
            return False

    def __or__(self, other):
        return self.logical_or(other)

    def logical_and(self, other):
        if isinstance(other, NumpyScalars):
            if self.isBool and other.isBool:
                return NumpyScalars(np.logical_and(self.v, other.v))
        else:
            if self.isBool:
                return NumpyScalars(np.logical_and(self.v, other))

    def logical_or(self, other):
        if isinstance(other, NumpyScalars):
            if self.isBool and other.isBool:
                return NumpyScalars(np.logical_or(self.v, other.v))
        else:
            if self.isBool:
                return NumpyScalars(np.logical_or(self.v, other))

    def logical_xor(self, other):
        if isinstance(other, NumpyScalars):
            if self.isBool and other.isBool:
                return NumpyScalars(np.logical_xor(self.v, other.v))
        else:
            if self.isBool:
                return NumpyScalars(np.logical_xor(self.v, other))

    def conditional_le(self, other, a, b):
        if isinstance(other, NumpyScalars):
            return NumpyScalars(np.where(self.v <= other, a, b))
        elif type(other) in (int, float):
            return NumpyScalars(np.where(self.v <= other, a, b))

    def conditional_lt(self, other, a, b):
        if isinstance(other, NumpyScalars):
            pass
        elif type(other) in (int, float):
            arr = np.where(self.v < other, a, b)
            return NumpyScalars(arr, ndtype=arr.dtype)

    def conditional_gt(self, other, a, b):
        if isinstance(other, NumpyScalars):
            pass
        elif type(other) in (int, float):
            arr = np.where(self.v > other, a, b)
            return NumpyScalars(arr, ndtype=arr.dtype)

    def conditional_ge(self, other, a, b):
        if isinstance(other, NumpyScalars):
            pass
        elif type(other) in (int, float):
            return NumpyScalars(np.where(self.v >= other, a, b))

    def conditional_eq(self, other, a, b):
        if isinstance(other, NumpyScalars):
            pass
        elif type(other) in (int, float):
            return NumpyScalars(np.where(self.v == other, a, b))

    def conditional_neq(self, other, a, b):
        if isinstance(other, NumpyScalars):
            pass
        elif type(other) in (int, float):
            return NumpyScalars(np.where(self.v != other, a, b))

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

    def __abs__(self):
        return Scalars(np.absolute(self.v))

    def all(self):
        return np.all(self.v)

    def any(self):
        return np.any(self.v)

    @property
    def isBool(self):
        return np.all(np.array_equal(self.v, self.v.astype(bool)))

    def toIntegers(self):
        return NumpyScalars(self.v.astype("int32"), ndtype="int32")


class CupyScalars:
    def __init__(self, array=None, N=None, ndtype="float64"):

        if array is not None:
            if type(array) == cp.ndarray:
                self.v = array.astype(ndtype)
            elif type(array) == cp.ndarray:
                self.v = array.astype(ndtype)
            else:
                self.v = cp.asarray(array, dtype=ndtype)

        elif N is not None:
            self.v = cp.zeros((N), dtype=ndtype)

        else:
            return

        N = len(self.v)
        self.binaryArrayTrue = cp.array([1] * N)
        self.binaryArrayFalse = cp.array([0] * N)

        self._iteration = 0

    def __repr__(self):
        return str(self.v)

    def __len__(self):
        return self.v.shape[0]

    def __add__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.add(self.v, other.v))
        elif isinstance(other, vec.CupyVectors):
            return ArithmeticError
        else:
            return CupyScalars(cp.add(self.v, other))

    def __sub__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.subtract(self.v, other.v))
        elif isinstance(other, vec.CupyVectors):
            return ArithmeticError
        elif type(other) in (float, int):
            return CupyScalars(cp.subtract(self.v, other))

    def __mul__(self, other):
        if isinstance(other, CupyScalars):
            if len(self) == len(other):
                return CupyScalars(cp.multiply(self.v, other.v))
        elif isinstance(other, vec.CupyVectors):
            if len(self) == len(other):
                return vec.CupyVectors(cp.multiply(self.v[:, None], other.v))
        elif type(other) in (float, int):
            return CupyScalars(cp.multiply(self.v, other))

    def __rmul__(self, other):
        if isinstance(other, CupyScalars):
            if len(self) == len(other):
                return CupyScalars(cp.multiply(self.v, other.v))
        elif isinstance(other, vec.CupyVectors):
            return ArithmeticError
        elif type(other) in (float, int):
            return CupyScalars(cp.multiply(self.v, other))

    def __truediv__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.true_divide(self.v, other.v))
        elif isinstance(other, vec.CupyVectors):
            return ArithmeticError
        else:
            return CupyScalars(cp.true_divide(self.v, other))

    def __neg__(self):
        return CupyScalars(cp.negative(self.v))

    def __invert__(self):
        return CupyScalars(cp.logical_not(self.v))

    def __getitem__(self, item):
        return CupyScalars(self.v[item])

    def __setitem__(self, key, value: cp.float32):
        self.v[key] = value

    def __eq__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v == other.v, self.binaryArrayTrue, self.binaryArrayFalse))
        else:
            return CupyScalars(cp.where(self.v == other, self.binaryArrayTrue, self.binaryArrayFalse))

    def __ne__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v != other.v, self.binaryArrayTrue, self.binaryArrayFalse))
        else:
            return CupyScalars(cp.where(self.v != other, self.binaryArrayTrue, self.binaryArrayFalse))

    def __lt__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v < other.v, self.binaryArrayTrue, self.binaryArrayFalse))
        else:
            return CupyScalars(cp.where(self.v < other, self.binaryArrayTrue, self.binaryArrayFalse))

    def __gt__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v > other.v, self.binaryArrayTrue, self.binaryArrayFalse))
        else:
            return CupyScalars(cp.where(self.v > other, self.binaryArrayTrue, self.binaryArrayFalse))

    def __le__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v <= other.v, self.binaryArrayTrue, self.binaryArrayFalse))
        else:
            return CupyScalars(cp.where(self.v <= other, self.binaryArrayTrue, self.binaryArrayFalse))

    def __ge__(self, other):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v >= other.v, self.binaryArrayTrue, self.binaryArrayFalse))
        else:
            return CupyScalars(cp.where(self.v >= other, self.binaryArrayTrue, self.binaryArrayFalse))

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

    def __contains__(self, item):
        if item in self.v:
            return True
        else:
            return False

    def __or__(self, other):
        return self.logical_or(other)

    def logical_and(self, other):
        if isinstance(other, CupyScalars):
            if self.isBool and other.isBool:
                return CupyScalars(cp.logical_and(self.v, other.v))
        else:
            if self.isBool:
                return CupyScalars(cp.logical_and(self.v, other))

    def logical_or(self, other):
        if isinstance(other, CupyScalars):
            if self.isBool and other.isBool:
                return CupyScalars(cp.logical_or(self.v, other.v))
        else:
            if self.isBool:
                return CupyScalars(cp.logical_or(self.v, other))

    def logical_xor(self, other):
        if isinstance(other, CupyScalars):
            if self.isBool and other.isBool:
                return CupyScalars(cp.logical_xor(self.v, other.v))
        else:
            if self.isBool:
                return CupyScalars(cp.logical_xor(self.v, other))

    def conditional_le(self, other, a, b):
        if isinstance(other, CupyScalars):
            return CupyScalars(cp.where(self.v <= other, a, b))
        elif type(other) in (int, float):
            return CupyScalars(cp.where(self.v <= other, a, b))

    def conditional_lt(self, other, a, b):
        if isinstance(other, CupyScalars):
            pass
        elif type(other) in (int, float):
            arr = cp.where(self.v < other, a, b)
            return CupyScalars(arr, ndtype=arr.dtype)

    def conditional_gt(self, other, a, b):
        if isinstance(other, CupyScalars):
            pass
        elif type(other) in (int, float):
            arr = cp.where(self.v > other, a, b)
            return CupyScalars(arr, ndtype=arr.dtype)

    def conditional_ge(self, other, a, b):
        if isinstance(other, CupyScalars):
            pass
        elif type(other) in (int, float):
            return CupyScalars(cp.where(self.v >= other, a, b))

    def conditional_eq(self, other, a, b):
        if isinstance(other, CupyScalars):
            pass
        elif type(other) in (int, float):
            return CupyScalars(cp.where(self.v == other, x=a, y=b))

    def conditional_neq(self, other, a, b):
        if isinstance(other, CupyScalars):
            pass
        elif type(other) in (int, float):
            return CupyScalars(cp.where(self.v != other, a, b))

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

    def all(self):
        return cp.all(self.v)

    def any(self):
        return cp.any(self.v)

    @property
    def isBool(self):
        return cp.all(cp.array_equal(self.v, self.v.astype(bool)))

    def toIntegers(self):
        return CupyScalars(self.v.astype("int32"), ndtype="int32")

    @property
    def isEmpty(self):
        if len(self.v) == 0:
            return True
        else:
            return False


Scalars = NumpyScalars