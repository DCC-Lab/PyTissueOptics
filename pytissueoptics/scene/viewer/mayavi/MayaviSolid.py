from typing import List, Tuple

from pytissueoptics.scene.solids import Solid


class MayaviSolid:
    def __init__(self, solid: 'Solid'):
        self._solid = solid
        self._primitive = solid.primitive
        self._x = []
        self._y = []
        self._z = []
        self._polygonIndices: List[Tuple[int]] = []

        self._create()

    @property
    def primitive(self) -> str:
        return self._primitive

    def _create(self):
        self._separateXYZ()
        self._findPolygonIndices()

    def _separateXYZ(self):
        for vertex in self._solid.vertices:
            self._x.append(vertex.x)
            self._y.append(vertex.y)
            self._z.append(vertex.z)

    def _findPolygonIndices(self):
        for surface in self._solid.surfaces:
            surfaceIndices = []
            for vertex in surface.vertices:
                index = self._solid.vertices.index(vertex)
                surfaceIndices.append(index)
            self._polygonIndices.append(tuple(surfaceIndices))

    @property
    def meshComponents(self):
        return self._x, self._y, self._z, self._polygonIndices
