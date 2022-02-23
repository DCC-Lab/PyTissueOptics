from typing import List, Tuple

from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer.mayavi import MayaviMesh
from pytissueoptics.scene.viewer.mayavi.MayaviNormals import MayaviNormals


class MayaviSolid:
    def __init__(self, solid: 'Solid', loadNormals=True):
        self._solid = solid
        self._primitive = solid.primitive
        self._x = []
        self._y = []
        self._z = []
        self._polygonIndices: List[Tuple[int]] = []
        self._loadNormals = loadNormals
        self._normals = MayaviNormals()

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
        for polygon in self._solid.getPolygons():
            polygonIndices = []
            for vertex in polygon.vertices:
                index = self._solid.vertices.index(vertex)
                polygonIndices.append(index)
            self._polygonIndices.append(tuple(polygonIndices))

            if self._loadNormals:
                self._normals.add(polygon.getCentroid(), polygon.normal)

    @property
    def mesh(self) -> MayaviMesh:
        return MayaviMesh(self._x, self._y, self._z, self._polygonIndices)

    @property
    def normals(self) -> MayaviNormals:
        return self._normals
