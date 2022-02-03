from typing import Dict, List

import numpy as np

from pytissueoptics.scene.geometry import Vector, Polygon, utils
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material


class Solid:
    def __init__(self, position: Vector, vertices: List[Vector], surfaces: Dict[str, List[Polygon]] = None,
                 material: Material = None, primitive: str = primitives.DEFAULT):
        self._material = material
        self._vertices = vertices
        self._surfaces = surfaces
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
    def vertices(self) -> List:
        return self._vertices

    @property
    def surfaces(self) -> List:
        surfaces = []
        for surfaceGroup in self._surfaces:
            for surface in self._surfaces[surfaceGroup]:
                surfaces.append(surface)
        return surfaces

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

        for surfaceGroup in self._surfaces.values():
            for surface in surfaceGroup:
                surface.resetNormal()

    @property
    def _verticesArray(self) -> np.ndarray:
        verticesArray = []
        for vertex in self._vertices:
            verticesArray.append(vertex.array)
        return np.asarray(verticesArray)


    def _computeMesh(self):
        self._surfaces = {}
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
        for surfaceGroup in self._surfaces.values():
            for surface in surfaceGroup:
                surface.insideMaterial = self._material

    def _setOutsideMaterial(self, material: Material, faceKey: str = None):
        if faceKey:
            for surface in self._surfaces[faceKey]:
                surface.outsideMaterial = material
        else:
            for surfaceGroup in self._surfaces.values():
                for surface in surfaceGroup:
                    surface.outsideMaterial = material
