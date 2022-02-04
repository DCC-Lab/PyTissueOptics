from typing import List, Tuple


class MayaviMesh:
    def __init__(self, x: List[float], y: List[float], z: List[float], polygonIndices: List[Tuple[int]])
        self._x = x
        self._y = y
        self._z = z
        self._polygonIndices = polygonIndices

    @property
    def components(self):
        return self._x, self._y, self._z, self._polygonIndices
