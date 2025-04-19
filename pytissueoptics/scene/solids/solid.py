import warnings
from typing import Callable, Dict, List

import numpy as np

from pytissueoptics.scene.geometry import (
    INTERFACE_KEY,
    BoundingBox,
    Environment,
    Polygon,
    Rotation,
    SurfaceCollection,
    Vector,
    Vertex,
    primitives,
    utils,
)

INITIAL_SOLID_ORIENTATION = Vector(0, 0, 1)


class Solid:
    def __init__(self, vertices: List[Vertex], position: Vector = Vector(0, 0, 0),
                 surfaces: SurfaceCollection = None, material=None,
                 label: str = "solid", primitive: str = primitives.DEFAULT, smooth: bool = False, labelOverride=True):
        self._vertices = vertices
        self._surfaces = surfaces
        self._material = material
        self._primitive = primitive
        self._position = Vector(0, 0, 0)
        self._rotation: Rotation = Rotation()
        self._orientation: Vector = INITIAL_SOLID_ORIENTATION
        self._bbox = None
        self._label = label
        self._layerLabels = {}

        if not self._surfaces:
            self._computeMesh()
        if labelOverride:
            self.setLabel(label)
        else:
            self._surfaces._solidLabel = ""

        self.translateTo(position)
        self._setInsideEnvironment()
        self._resetBoundingBoxes()
        self._resetPolygonsCentroids()

        self._smoothing = False
        self._setVertexNormals()
        if smooth:
            self.smooth()

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def vertices(self) -> List[Vertex]:
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

    def getVertices(self) -> List[Vertex]:
        return self.vertices

    def getLabel(self) -> str:
        return self._label

    def setLabel(self, label: str):
        self._surfaces.updateSolidLabel(label)
        self._label = label

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

    def scale(self, factor: float):
        for v in self._vertices:
            v.multiply(factor)

        previousPosition = self._position.copy()
        self._position = self._position * factor
        positionDifference = previousPosition - self._position
        self.translateBy(positionDifference)

    def rotate(self, xTheta=0, yTheta=0, zTheta=0, rotationCenter: Vector = None):
        """
        Requires the angle in degrees for each axis around which the solid will be rotated.

        Since we know the position of the centroid in global coordinates, we extract a centered array reference
        to the vertices and rotate them with euler rotation before moving that reference back to the solid's position.
        Finally, we update the solid vertices' components with the values of this rotated array reference and ask each
        solid surface to compute its new normal.
        """
        rotation = Rotation(xTheta, yTheta, zTheta)
        rotationFunction = lambda vertices: self._rotateWithEuler(vertices, rotation)

        self._rotateWith(rotationFunction, rotationCenter)
        self._rotation.add(rotation)

    def orient(self, towards: Vector):
        """ Rotate the solid so that its direction is aligned with the given vector "towards". 
        Note that the original solid orientation is set to (0, 0, 1). """
        initialOrientation = self._orientation
        axis, angle = utils.getAxisAngleBetween(initialOrientation, towards)
        rotationFunction = lambda vertices: self._rotateWithAxisAngle(vertices, axis, angle)

        self._rotateWith(rotationFunction, None)
        self._orientation = towards

    def _rotateWith(self, rotationFunction: Callable[[List[Vector]], List[Vector]], rotationCenter: Vector = None):
        if rotationCenter is None:
            rotationCenter = self.position
        verticesAtOrigin: List[Vector] = [vertex - rotationCenter for vertex in self._vertices]
        verticesAtOrigin.append(self.position - rotationCenter)

        rotatedVerticesAtOrigin = rotationFunction(verticesAtOrigin)

        rotatedVertices = [vertex + rotationCenter for vertex in rotatedVerticesAtOrigin]

        for (vertex, rotatedVertex) in zip(self._vertices, rotatedVertices):
            vertex.update(*rotatedVertex.array)

        self._position = rotatedVertices[-1]

        self._surfaces.resetNormals()
        self._resetBoundingBoxes()
        self._resetPolygonsCentroids()

        if self._smoothing:
            self.smooth()

    @staticmethod
    def _rotateWithAxisAngle(vertices: List[Vector], axis: Vector, angle: float) -> List[Vector]:
        for vertex in vertices:
            vertex.rotateAround(axis, angle)
        return vertices

    @staticmethod
    def _rotateWithEuler(vertices: List[Vector], rotation: Rotation, inverse: bool = False) -> List[Vector]:
        verticesArray = np.asarray([vertex.array for vertex in vertices])
        rotatedVerticesArray = utils.rotateVerticesArray(verticesArray, rotation, inverse)
        return [Vector(*vertex) for vertex in rotatedVerticesArray]

    def getEnvironment(self, surfaceLabel: str = None) -> Environment:
        if surfaceLabel:
            return self.surfaces.getInsideEnvironment(surfaceLabel)
        else:
            return Environment(self._material, self)

    def setOutsideEnvironment(self, environment: Environment, surfaceLabel: str = None):
        self._surfaces.setOutsideEnvironment(environment, surfaceLabel)

    @property
    def surfaceLabels(self) -> List[str]:
        return self._surfaces.surfaceLabels

    def getPolygons(self, surfaceLabel: str = None) -> List[Polygon]:
        return self._surfaces.getPolygons(surfaceLabel)

    def setPolygons(self, surfaceLabel: str, polygons: List[Polygon]):
        self._surfaces.setPolygons(surfaceLabel, polygons)

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

    def _setInsideEnvironment(self):
        polygons = self._surfaces.getPolygons()
        if not self._material and polygons[0].insideEnvironment is not None:
            return
        for polygon in polygons:
            polygon.setInsideEnvironment(Environment(self._material, self))

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

    def contains(self, *vertices: Vector) -> bool:
        """
        Provides a simple implementation, which should be overwritten by subclasses to provide more accuracy.
        This implementation will only check the outer bounding box of the solid to check if the vertices are outside. 
        Similarly, it will create a max internal bounding box and check if the vertices are inside. 
        """
        for vertex in vertices:
            if not self._bbox.contains(vertex):
                return False
        internalBBox = self._getInternalBBox()
        for vertex in vertices:
            if not internalBBox.contains(vertex):
                warnings.warn(f"Method contains(Vertex) is not implemented for Solids of type {type(self).__name__}. "
                              "Returning False since Vertex does not lie in the internal bounding box "
                              "(underestimating containment). ", RuntimeWarning)
                return False
        return True

    def _getInternalBBox(self):
        insideBBox = self._bbox.copy()
        for polygon in self.getPolygons():
            insideBBox.exclude(polygon.bbox)
        return insideBBox

    def _applyInverseRotation(self, vertices: List[Vector]) -> List[Vector]:
        if self._rotation and self._orientation != INITIAL_SOLID_ORIENTATION:
            raise Exception("Rotation correction (often used for solid containment checks) "
                            "is not implemented for solids that underwent rotations "
                            "with both the Euler rotate() and the axis-angle orient() methods.")
        if self._rotation:
            return self._rotateWithEuler(vertices, self._rotation, inverse=True)
        if self._orientation != INITIAL_SOLID_ORIENTATION:
            axis, angle = utils.getAxisAngleBetween(self._orientation, INITIAL_SOLID_ORIENTATION)
            return self._rotateWithAxisAngle(vertices, axis, angle)
        return vertices

    def isStack(self) -> bool:
        for surfaceLabel in self.surfaceLabels:
            if INTERFACE_KEY in surfaceLabel:
                return True
        return False

    def getLayerLabelMap(self) -> Dict[str, List[str]]:
        return self._layerLabels

    def getLayerLabels(self) -> List[str]:
        return list(self._layerLabels.keys())

    def getLayerSurfaceLabels(self, layerSolidLabel) -> List[str]:
        return list(self._layerLabels[layerSolidLabel])

    def completeSurfaceLabel(self, surfaceLabel: str) -> str:
        return self._surfaces.processLabel(surfaceLabel)

    def smooth(self, surfaceLabel: str = None, reset: bool = True):
        """ Prepare smoothing by calculating vertex normals. This is not done
        by default. The vertex normals are used during ray-polygon intersection
        to return an interpolated (smooth) normal. A vertex normal is defined
        by taking the average normal of all adjacent polygons.

        This base implementation will smooth all surfaces by default. This can
        be changed by overwriting the signature with a specific surfaceLabel in
        another solid implementation and calling super().smooth(surfaceLabel).
        """
        self._setVertexNormals(surfaceLabel, smooth=True, reset=reset)

    def _setVertexNormals(self, surfaceLabel: str = None, smooth=False, reset=True):
        self._smoothing = smooth
        if reset:
            for vertex in self.vertices:
                vertex.normal = None

        polygons = self.getPolygons(surfaceLabel)

        for polygon in polygons:
            polygon.toSmooth = smooth
            for vertex in polygon.vertices:
                if vertex.normal:
                    vertex.normal += polygon.normal
                else:
                    vertex.normal = polygon.normal.copy()

        for vertex in self.vertices:
            if vertex.normal:
                vertex.normal.normalize()

    def __hash__(self):
        verticesHash = hash(tuple(sorted([hash(v) for v in self._vertices])))
        materialHash = hash(self._material) if self._material else 0
        return hash((verticesHash, materialHash))
