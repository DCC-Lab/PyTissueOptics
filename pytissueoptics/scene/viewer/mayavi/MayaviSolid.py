from typing import List, Tuple

from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer.mayavi import MayaviTriangleMesh
from pytissueoptics.scene.viewer.mayavi.MayaviNormals import MayaviNormals


class MayaviSolid:
    def __init__(self, solid: 'Solid', loadNormals=True):
        self._solid = solid
        self._primitive = solid.primitive
        self._x = []
        self._y = []
        self._z = []
        self._polygonsIndices: List[Tuple[int]] = []
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
            self._polygonsIndices.append(tuple(polygonIndices))

            if self._loadNormals:
                self._normals.add(polygon.getCentroid(), polygon.normal)

    @property
    def triangleMesh(self) -> MayaviTriangleMesh:
        return MayaviTriangleMesh(self._x, self._y, self._z, self._getTriangleIndices())

    def _getTriangleIndices(self):
        if self._solid.primitive == primitives.TRIANGLE:
            return self._polygonsIndices
        else:
            trianglesIndices = []
            for polygonIndices in self._polygonsIndices:
                for i in range(len(polygonIndices) - 2):
                    trianglesIndices.append((polygonIndices[0],
                                             polygonIndices[i + 1],
                                             polygonIndices[i + 2]))
            return trianglesIndices

    @property
    def normals(self) -> MayaviNormals:
        return self._normals
