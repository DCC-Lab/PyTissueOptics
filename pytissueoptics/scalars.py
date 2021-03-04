import numpy as np
from .vectors import *

class Scalars:
    """ An array of scalars that is compatible with operations on Vectors 
    There is a reason for not using numpy.array directly: we want to
    add new functions that will be specific to our problem here,
    and Python does not allow us to extend system classes.

    The class works with float, int and bool.  With boolean values, True is 1.
    """
    def __init__(self, array=None, N=None):
        if array is not None:
            self.v = np.array(array)
        elif N is not None:
            self.v = np.array([0]*N)

        self._iteration = 0

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

    def __len__(self):
        return len(self.v)

    def __mul__(self, scale):
        return Scalars(self.v*scale)

    def __rmul__(self, scale):
        return Scalars(self.v*scale)

    def __truediv__(self, scale):
        return Scalars(self.v/scale)

    def __add__(self, rhs):
        return Scalars([v1+v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __neg__(self):
        return Scalars([-v1 for v1 in self.v])

    def __sub__(self, rhs):
        return Scalars([v1-v2 for (v1, v2) in list(zip(self.v, rhs.v))])

    def __getitem__(self, index):
        return self.v[index]

    def __setitem__(self, index, newvalue): 
        self.v[index] = newvalue

    def __eq__(self, rhs):
        if isinstance(rhs, Scalars):
            each = [v1 == v2 for (v1, v2) in list(zip(self.v, rhs.v))]
        else:
            each = [v1 == v2 for (v1, v2) in list(zip(self.v, rhs))]            
        return np.array(each).all()

class NumpyScalars:
    def __init__(self, count: int, value: float):
        self.s = np.full((1, count), value, dtype=np.float32)

    def __add__(self, other):
        if isinstance(other, NumpyScalars):
            result = np.add(self.s, other.s)
            return result

    def __sub__(self, other):
        if isinstance(other, NumpyScalars):
            result = np.subtract(self.s, other.s)
            return result

    def __mul__(self, other):
        if isinstance(other, NumpyScalars):
            result = np.multiply(self.s, other.s)
            return result
        elif isinstance(other, NumpyVectors):
            result = np.multiply(self.s[:, None] * other.v)
            return result
        else:
            raise NotImplemented

    def __truediv__(self, other):
        pass

    def __repr__(self):
        pass

    def __neg__(self):
        result = np.negative(self.s)
        return result

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def __eq__(self, other):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    @classmethod
    def random(cls, count: int):
        s = np.random.rand(1, count)
        return s

    @classmethod
    def random2(cls, count: int):
        s = (np.random.rand(1, count) * 2) - 1
        return s
