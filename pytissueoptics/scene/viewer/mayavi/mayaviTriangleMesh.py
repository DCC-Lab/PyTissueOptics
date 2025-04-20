from typing import List, Tuple


class MayaviTriangleMesh:
    def __init__(self, x: List[float], y: List[float], z: List[float], triangleIndices: List[Tuple[int]]):
        self._x = x
        self._y = y
        self._z = z
        self._triangleIndices = triangleIndices

    @property
    def components(self):
        return self._x, self._y, self._z, self._triangleIndices
