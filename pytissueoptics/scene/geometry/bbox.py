from typing import List

from pytissueoptics.scene.geometry import Vector, Vertex


class BoundingBox:
    _AXIS_KEYS = ['x', 'y', 'z']
    _LIMIT_KEYS = ['min', 'max']

    def __init__(self, xLim: List[float], yLim: List[float], zLim: List[float], validate=True):
        self._xLim = xLim
        self._yLim = yLim
        self._zLim = zLim
        self._xyzLimits = [self._xLim, self._yLim, self._zLim]
        if validate:
            self._checkIfCoherent()

    def __repr__(self) -> str:
        return f"<BoundingBox>:(xLim={self._xLim}, yLim={self._yLim}, zLim={self._zLim})"

    def __eq__(self, other: 'BoundingBox') -> bool:
        if self._xLim == other._xLim and self._yLim == other._yLim and self._zLim == other._zLim:
            return True
        else:
            return False

    def copy(self):
        return BoundingBox([self._xLim[0], self._xLim[1]], [self._yLim[0], self._yLim[1]], [self._zLim[0], self._zLim[1]])

    def _checkIfCoherent(self):
        if not (self.xMax >= self.xMin and self.yMax >= self.yMin and self.zMax >= self.zMin):
            raise ValueError("Maximum limit value cannot be lower than minimum limit value.")

    @classmethod
    def fromVertices(cls, vertices: List[Vertex]) -> 'BoundingBox':
        vertexIter = range(len(vertices))
        x = sorted([vertices[i].x for i in vertexIter])
        y = sorted([vertices[i].y for i in vertexIter])
        z = sorted([vertices[i].z for i in vertexIter])
        xLim = [x[0], x[-1]]
        yLim = [y[0], y[-1]]
        zLim = [z[0], z[-1]]
        return BoundingBox(xLim, yLim, zLim, validate=False)

    @classmethod
    def fromPolygons(cls, polygons: List['Polygon']) -> 'BoundingBox':
        bbox = None
        for polygon in polygons:
            if bbox is not None:
                bbox.extendTo(polygon.bbox)
            else:
                bbox = polygon.bbox.copy()
        return bbox

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

    @property
    def xyzLimits(self) -> List[List[float]]:
        return self._xyzLimits

    def getAxisWidth(self, axis: str) -> float:
        if axis == "x":
            return self.xWidth
        elif axis == "y":
            return self.yWidth
        elif axis == "z":
            return self.zWidth

    def getAxisLimit(self, axis: str, limit: str) -> float:
        return self._xyzLimits[self._AXIS_KEYS.index(axis)][self._LIMIT_KEYS.index(limit)]

    def getAxisLimits(self, axis: str) -> List[float]:
        return self._xyzLimits[self._AXIS_KEYS.index(axis)]

    def update(self, axis: str, limit: str, value: float):
        self._xyzLimits[self._AXIS_KEYS.index(axis)][self._LIMIT_KEYS.index(limit)] = value
        self._checkIfCoherent()

    def getArea(self):
        a = self.xWidth
        b = self.yWidth
        c = self.zWidth
        return a * b * 2 + a * c * 2 + b * c * 2

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

    def shrinkTo(self, other: 'BoundingBox'):
        if other.xMin > self.xMin:
            self._xLim[0] = other.xMin
        if other.xMax < self.xMax:
            self._xLim[1] = other.xMax
        if other.yMin > self.yMin:
            self._yLim[0] = other.yMin
        if other.yMax < self.yMax:
            self._yLim[1] = other.yMax
        if other.zMin > self.zMin:
            self._zLim[0] = other.zMin
        if other.zMax < self.zMax:
            self._zLim[1] = other.zMax

    def exclude(self, other: 'BoundingBox'):
        if not self.intersects(other):
            return

        largestArea = 0
        currentBBox = None
        for axis in range(3):
            leftLimits = [self[axis][0], other[axis][0]]
            rightLimits = [other[axis][1], self[axis][1]]
            leftLength = leftLimits[1] - leftLimits[0]
            rightLength = rightLimits[1] - rightLimits[0]
            croppedLimits = leftLimits if leftLength > rightLength else rightLimits
            newXYZLimits = self._xyzLimits.copy()
            newXYZLimits[axis] = croppedLimits
            newBBox = BoundingBox(*newXYZLimits)
            boxArea = newBBox.getArea()
            if boxArea > largestArea:
                largestArea = boxArea
                currentBBox = newBBox

        for axis in range(3):
            self._xyzLimits[axis][0] = currentBBox[axis][0]
            self._xyzLimits[axis][1] = currentBBox[axis][1]

    def __getitem__(self, index: int) -> List[float]:
        return self._xyzLimits[index]

    def intersects(self, other: 'BoundingBox') -> bool:
        for axis in range(3):
            if other[axis][0] > self[axis][1] or other[axis][1] < self[axis][0]:
                return False
        return True
