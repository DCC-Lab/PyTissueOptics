from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vector, Quad, Triangle, Vertex, BoundingBox, Polygon, primitives
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import SurfaceCollection
from pytissueoptics.scene.solids.stack.cuboidStacker import CuboidStacker
from pytissueoptics.scene.solids.stack.stackResult import StackResult


class Cuboid(Solid):
    """
        Also known as the Right Rectangular Prism, the Cuboid is defined by its
        width (a, b, c) in each axis (x, y, z) respectively.

        The position refers to the vector from global origin to its centroid.
        The generated mesh will be divided into the following subgroups:
        Left (-x), Right (+x), Bottom (-y), Top (+y), Front (-z), Back (+z).
    """

    def __init__(self, a: float, b: float, c: float,
                 vertices: List[Vertex] = None, position: Vector = Vector(0, 0, 0), surfaces: SurfaceCollection = None,
                 material=None, label: str = "cuboid", primitive: str = primitives.DEFAULT, labelOverride=True):

        self.shape = [a, b, c]

        if not vertices:
            vertices = [Vertex(-a / 2, -b / 2, c / 2), Vertex(a / 2, -b / 2, c / 2), Vertex(a / 2, b / 2, c / 2),
                        Vertex(-a / 2, b / 2, c / 2),
                        Vertex(-a / 2, -b / 2, -c / 2), Vertex(a / 2, -b / 2, -c / 2), Vertex(a / 2, b / 2, -c / 2),
                        Vertex(-a / 2, b / 2, -c / 2)]

        super().__init__(vertices, position, surfaces, material, label, primitive, labelOverride=labelOverride)

    def _computeTriangleMesh(self):
        V = self._vertices
        self._surfaces.add('left', [Triangle(V[4], V[0], V[3]), Triangle(V[3], V[7], V[4])])
        self._surfaces.add('right', [Triangle(V[1], V[5], V[6]), Triangle(V[6], V[2], V[1])])
        self._surfaces.add('bottom', [Triangle(V[4], V[5], V[1]), Triangle(V[1], V[0], V[4])])
        self._surfaces.add('top', [Triangle(V[3], V[2], V[6]), Triangle(V[6], V[7], V[3])])
        self._surfaces.add('front', [Triangle(V[5], V[4], V[7]), Triangle(V[7], V[6], V[5])])
        self._surfaces.add('back', [Triangle(V[0], V[1], V[2]), Triangle(V[2], V[3], V[0])])

    def _computeQuadMesh(self):
        V = self._vertices
        self._surfaces.add('left', [Quad(V[4], V[0], V[3], V[7])])
        self._surfaces.add('right', [Quad(V[1], V[5], V[6], V[2])])
        self._surfaces.add('bottom', [Quad(V[4], V[5], V[1], V[0])])
        self._surfaces.add('top', [Quad(V[3], V[2], V[6], V[7])])
        self._surfaces.add('front', [Quad(V[5], V[4], V[7], V[6])])
        self._surfaces.add('back', [Quad(V[0], V[1], V[2], V[3])])

    def stack(self, other: 'Cuboid', onSurface: str = 'top', stackLabel="CuboidStack") -> 'Cuboid':
        """
        Basic implementation for stacking cuboids along an axis.

        For example, stacking on 'top' will move the other cuboid on top of this cuboid. They will now share
         the same mesh at the interface and inside/outside materials at the interface will be properly defined.
         This will return a new cuboid that represents the stack, with a new 'interface<i>' surface group.

        Limitations:
            - Requires cuboids with the same shape except along the stack axis.
            - Cannot stack another stack unless it is along its stacked axis (ill-defined interface material).
            - Expected behavior not guaranteed for pre-rotated cuboids.
            - Stacked cuboids will lose reference to their initial stack surface (not a closed solid anymore).
                Use the returned cuboid stack.
        """
        stacker = CuboidStacker()
        stackResult = stacker.stack(onCuboid=self, otherCuboid=other, onSurface=onSurface)
        return Cuboid._fromStackResult(stackResult, label=stackLabel)

    @classmethod
    def _fromStackResult(cls, stackResult: StackResult, label: str) -> 'Cuboid':
        # subtracting stackCentroid from all vertices because solid creation will translate back to position.
        for vertex in stackResult.vertices:
            vertex.subtract(stackResult.position)

        cuboid = Cuboid(*stackResult.shape, position=stackResult.position, vertices=stackResult.vertices,
                        surfaces=stackResult.surfaces, label=label, primitive=stackResult.primitive, labelOverride=False)
        cuboid._layerLabels = stackResult.layerLabels
        return cuboid

    def contains(self, *vertices: Vector) -> bool:
        relativeVertices = [vertex - self.position for vertex in vertices]
        relativeVertices = self._applyInverseRotation(relativeVertices)

        relativeVertices = np.asarray([vertex.array for vertex in relativeVertices])
        bounds = [s/2 for s in self.shape]
        if np.any(np.abs(relativeVertices) >= bounds):
            return False

        if len(vertices) == 1:
            return True

        if self.isStack():
            # At this point, all vertices are contained in the bounding box of the stack.
            # We just need to make sure they are contained in a single layer.
            return self._aSingleLayerContains(relativeVertices)
        return True

    def _aSingleLayerContains(self, relativeVertices: np.ndarray) -> bool:
        for layerLabel in self._layerLabels:
            bbox = self._getLayerBBox(layerLabel)
            bboxShape = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
            bounds = [s/2 for s in bboxShape]
            layerOffset = bbox.center - self.position
            layerRelativeVertices = relativeVertices - np.asarray(layerOffset.array)
            if np.all(np.abs(layerRelativeVertices) < bounds):
                return True
        return False

    def _getLayerBBox(self, layerLabel: str) -> BoundingBox:
        polygons = self._getLayerPolygons(layerLabel)
        bbox = BoundingBox.fromPolygons(polygons)
        return bbox

    def _getLayerPolygons(self, layerLabel: str) -> List[Polygon]:
        layerSurfaceLabels = self._layerLabels[layerLabel]
        polygons = []
        for surfaceLabel in layerSurfaceLabels:
            polygons.extend(self._surfaces.getPolygons(surfaceLabel))
        return polygons
