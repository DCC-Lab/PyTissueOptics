from typing import Dict, List

from python_graphics_engine.geometry import Vector, Polygon, primitives
from python_graphics_engine.materials import Material


class Solid:
    def __init__(self, position: Vector, material: Material,
                 vertices: List[Vector], surfaces: Dict[str, List[Polygon]],
                 primitive: str = primitives.DEFAULT):
        self._material = material
        self._vertices = vertices
        self._surfaces = surfaces
        self._primitive = primitive
        self._position = Vector(0, 0, 0)

        self.translateTo(position)
        self._computeMesh()
        self._setInsideMaterial()

    @property
    def position(self) -> Vector:
        return self._position

    def translateTo(self, position):
        if position == self._position:
            return
        translationVector = position - self._position
        self.translateBy(translationVector)

    def translateBy(self, translationVector: Vector):
        self._position.add(translationVector)
        for v in self._vertices:
            v.add(translationVector)

    def _computeMesh(self):
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
