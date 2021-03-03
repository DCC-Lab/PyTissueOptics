import numpy as np


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
