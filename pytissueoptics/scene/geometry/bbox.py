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

    def __eq__(self, other: 'BoundingBox') -> bool:
        if self._xLim == other._xLim and self._yLim == other._yLim and self._zLim == other._zLim:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.xMin, self.xMax, self.yMin, self.yMax, self.zMin, self.zMax))

    def _checkIfCoherent(self):
        if not (self.xMax >= self.xMin and self.yMax >= self.yMin and self.zMax >= self.zMin):
            print("Oh-Oh")

    @classmethod
    def fromVertices(cls, vertices: List[Vector]) -> 'BoundingBox':
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

    @property
    def xWidth(self):
        return self.xMax - self.xMin

    @property
    def yWidth(self):
        return self.yMax - self.yMin

    @property
    def zWidth(self):
        return self.zMax - self.zMin

    def getAxisWidth(self, axis: str) -> float:
        if axis == "x":
            return self.xWidth
        elif axis == "y":
            return self.yWidth
        elif axis == "z":
            return self.zWidth

    def getAxisLimit(self, axis: str, limit: str) -> float:
        return self._xyzLimits[self._axisKeys.index(axis)][self._limitKeys.index(limit)]

    def getAxisLimits(self, axis: str) -> List[float]:
        return self._xyzLimits[self._axisKeys.index(axis)]

    def change(self, axis: str, limit: str, value: float):
        self._xyzLimits[self._axisKeys.index(axis)][self._limitKeys.index(limit)] = value
        self._checkIfCoherent()

    def changeToNew(self, axis: str, limit: str, value: float) -> 'BoundingBox':
        newBbox = deepcopy(self)
        newBbox.change(axis, limit, value)
        newBbox._checkIfCoherent()
        return newBbox

    def newFrom(self):
        newBbox = deepcopy(self)
        return newBbox

    def getArea(self):
        a = self.xWidth
        b = self.yWidth
        c = self.zWidth
        return a*b*2 + a*c*2 + b*c*2

    def contains(self, point: Vector):
        xCondition = self.xMin < point.x < self.xMax
        yCondition = self.yMin < point.y < self.yMax
        zCondition = self.zMin < point.z < self.zMax
        if xCondition and yCondition and zCondition:
            return True
        else:
            return False

    def extendTo(self, other: 'BoundingBox'):
        if other.xMin < self.xMin:
            self._xLim[0] = other.xMin
        if other.xMax > self.xMax:
            self._xLim[1] = other.xMax
        if other.yMin < self.yMin:
            self._yLim[0] = other.yMin
        if other.yMax > self.yMax:
            self._yLim[1] = other.yMax
        if other.zMin < self.zMin:
            self._zLim[0] = other.zMin
        if other.zMax > self.zMax:
            self._zLim[1] = other.zMax

    def __getitem__(self, index: int) -> List[float]:
        return self._xyzLimits[index]
