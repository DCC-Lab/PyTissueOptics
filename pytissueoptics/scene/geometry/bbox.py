from __future__ import annotations
from typing import List
from copy import deepcopy
from pytissueoptics.scene.geometry import Vector


class BoundingBox:
    def __init__(self, xLim: List[float], yLim: List[float], zLim: List[float]):
        self._axisKeys = ["x", "y", "z"]
        self._limitKeys = ["min", "max"]
        self._xLim = xLim
        self._yLim = yLim
        self._zLim = zLim
        self._xyzLimits = [xLim, yLim, zLim]
        self._checkIfCoherent()

    def __repr__(self) -> str:
        return str([self._xLim, self._yLim, self._zLim])

    def __eq__(self, other: BoundingBox) -> bool:
        if self._xLim == other._xLim and self._yLim == other._yLim and self._zLim == other._zLim:
            return True
        else:
            return False

    def _checkIfCoherent(self):
        if self.xMax > self.xMin and self.yMax > self.yMin and self.zMax > self.zMin:
            return True
        else:
            raise ValueError("Maximum limit value cannot be lower than minimum limit value.")

    @classmethod
    def fromVertices(cls, vertices: List[Vector]) -> BoundingBox:
        x = [vertices[i].x for i in range(len(vertices))]
        y = [vertices[i].y for i in range(len(vertices))]
        z = [vertices[i].z for i in range(len(vertices))]
        xLim = [min(x), max(x)]
        yLim = [min(y), max(y)]
        zLim = [min(z), max(z)]
        return BoundingBox(xLim, yLim, zLim)

    @property
    def xMin(self) -> float:
        return self._xLim[0]

    @property
    def xMax(self) -> float:
        return self._xLim[1]

    @property
    def yMin(self) -> float:
        return self._yLim[0]

    @property
    def yMax(self) -> float:
        return self._yLim[1]

    @property
    def zMin(self) -> float:
        return self._zLim[0]

    @property
    def zMax(self) -> float:
        return self._zLim[1]

    @property
    def center(self) -> Vector:
        return Vector(self.xMin + (self.xMax - self.xMin) / 2, self.yMin + (self.yMax - self.yMin) / 2,
                      self.zMin + (self.zMax - self.zMin) / 2)

    @property
    def xLim(self) -> List[float]:
        return self._xLim

    @property
    def yLim(self) -> List[float]:
        return self._yLim

    @property
    def zLim(self) -> List[float]:
        return self._zLim

    def __getitem__(self, index: int) -> List[float]:
        return self._xyzLimits[index]

    def change(self, axis="x", limit="max", value=0):
        self._xyzLimits[self._axisKeys.index(axis)][self._axisKeys.index(limit)] = value

    def changeToNew(self, axis="x", limit="max", value=0) -> BoundingBox:
        newBbox = deepcopy(self)
        newBbox.change(axis, limit, value)
        return newBbox
