from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vector, utils, Polygon, Rotation, BoundingBox
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.geometry import SurfaceCollection


class Solid:
    def __init__(self, vertices: List[Vector], position: Vector = Vector(0, 0, 0),
                 surfaces: SurfaceCollection = None, material: Material = None, primitive: str = primitives.DEFAULT):
        self._vertices = vertices
        self._surfaces = surfaces
        self._material = material
        self._primitive = primitive
        self._position = Vector(0, 0, 0)
        self._orientation: Rotation = Rotation()
        self._bbox = None

        if not self._surfaces:
            self._computeMesh()

        self.translateTo(position)
        self._setInsideMaterial()
        self._resetBoundingBoxes()
        self._resetPolygonsCentroids()

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

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox

    def getBoundingBox(self) -> BoundingBox:
        return self.bbox

    def getVertices(self) -> List[Vector]:
        return self.vertices

    def _resetBoundingBoxes(self):
        self._bbox = BoundingBox.fromVertices(self._vertices)
        self._surfaces.resetBoundingBoxes()

    def _resetPolygonsCentroids(self):
        self._surfaces.resetCentroids()

    def translateTo(self, position):
        if position == self._position:
            return
        translationVector = position - self._position
        self.translateBy(translationVector)

    def translateBy(self, translationVector: Vector):
        self._position.add(translationVector)
        for v in self._vertices:
            v.add(translationVector)
        self._resetBoundingBoxes()
        self._resetPolygonsCentroids()

    def rotate(self, xTheta=0, yTheta=0, zTheta=0):
        """
        Requires the angle in degrees for each axis around which the solid will be rotated.

        Since we know the position of the centroid in global coordinates, we extract a centered array reference
        to the vertices and rotate them with euler rotation before moving that reference back to the solid's position.
        Finally we update the solid vertices' components with the values of this rotated array reference and ask each
        solid surface to compute its new normal.
        """
        rotation = Rotation(xTheta, yTheta, zTheta)

        verticesArrayAtOrigin = self._verticesArray - self.position.array
        rotatedVerticesArrayAtOrigin = utils.rotateVerticesArray(verticesArrayAtOrigin, rotation)
        rotatedVerticesArray = rotatedVerticesArrayAtOrigin + self.position.array

        for (vertex, rotatedVertexArray) in zip(self._vertices, rotatedVerticesArray):
            vertex.update(*rotatedVertexArray)

        self._orientation.add(rotation)
        self._surfaces.resetNormals()
        self._resetBoundingBoxes()
        self._resetPolygonsCentroids()

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

        currentVerticesIDs = {id(vertex) for vertex in self._vertices}
        newVertices = []
        for polygon in polygons:
            newVertices.extend(polygon.vertices)
        for vertex in newVertices:
            if id(vertex) not in currentVerticesIDs:
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

    def contains(self, *vertices: Vector) -> bool:
        raise NotImplementedError

    def isStack(self) -> bool:
        for surfaceName in self.surfaceNames:
            if "Interface" in surfaceName:
                return True
        return False
