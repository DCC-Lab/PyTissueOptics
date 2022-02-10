from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vector, utils, Polygon
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids.surfaceCollection import SurfaceCollection


class Solid:
    def __init__(self, vertices: List[Vector], position: Vector = Vector(0, 0, 0),
                 surfaces: SurfaceCollection = None, material: Material = None, primitive: str = primitives.DEFAULT):
        self._vertices = vertices
        self._surfaces = surfaces
        self._material = material
        self._primitive = primitive
        self._position = Vector(0, 0, 0)

        if not self._surfaces:
            self._computeMesh()

        self.translateTo(position)
        self._setInsideMaterial()

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def vertices(self) -> List[Vector]:
        return self._vertices

    @property
    def surfaces(self) -> SurfaceCollection:
        return self._surfaces

    @property
    def primitive(self) -> str:
        return self._primitive

    def translateTo(self, position):
        if position == self._position:
            return
        translationVector = position - self._position
        self.translateBy(translationVector)

    def translateBy(self, translationVector: Vector):
        self._position.add(translationVector)
        for v in self._vertices:
            v.add(translationVector)
    
    def rotate(self, xTheta=0, yTheta=0, zTheta=0):
        """
        Requires the angle in degrees for each axis around which the solid will be rotated.

        Since we know the position of the centroid in global coordinates, we extract a centered array reference
        to the vertices and rotate them with euler rotation before moving that reference back to the solid's position.
        Finally we update the solid vertices' components with the values of this rotated array reference and ask each
        solid surface to compute its new normal.
        """
        verticesArrayAtOrigin = self._verticesArray - self.position.array
        rotatedVerticesArrayAtOrigin = utils.rotateVerticesArray(verticesArrayAtOrigin, xTheta, yTheta, zTheta)
        rotatedVerticesArray = rotatedVerticesArrayAtOrigin + self.position.array

        for (vertex, rotatedVertexArray) in zip(self._vertices, rotatedVerticesArray):
            vertex.update(*rotatedVertexArray)

        self._surfaces.resetNormals()

    def getMaterial(self, surfaceName: str = None) -> Material:
        if surfaceName:
            return self.surfaces.getInsideMaterial(surfaceName)
        else:
            return self._material

    def setOutsideMaterial(self, material: Material, surfaceName: str = None):
        self._surfaces.setOutsideMaterial(material, surfaceName)

    @property
    def surfaceNames(self) -> List[str]:
        return self._surfaces.surfaceNames

    def getPolygons(self, surfaceName: str = None) -> List[Polygon]:
        return self._surfaces.getPolygons(surfaceName)

    def setPolygons(self, surfaceName: str, polygons: List[Polygon]):
        self._surfaces.setPolygons(surfaceName, polygons)

        newVertices = []
        for polygon in polygons:
            newVertices.extend(polygon.vertices)
        for vertex in newVertices:
            if vertex not in self._vertices:
                self._vertices.append(vertex)

    @property
    def _verticesArray(self) -> np.ndarray:
        verticesArray = []
        for vertex in self._vertices:
            verticesArray.append(vertex.array)
        return np.asarray(verticesArray)

    def _computeMesh(self):
        self._surfaces = SurfaceCollection()
        if self._primitive == primitives.TRIANGLE:
            self._computeTriangleMesh()
        elif self._primitive == primitives.QUAD:
            self._computeQuadMesh()
        else:
            raise NotImplementedError(f"Solid mesh not implemented for primitive '{self._primitive}'")

    def _computeTriangleMesh(self):
        raise NotImplementedError(f"Triangle mesh not implemented for Solids of type {type(self).__name__}")

    def _computeQuadMesh(self):
        raise NotImplementedError(f"Quad mesh not implemented for Solids of type {type(self).__name__}")

    def _setInsideMaterial(self):
        if self._material is None:
            return
        for polygon in self._surfaces.getPolygons():
            polygon.setInsideMaterial(self._material)
