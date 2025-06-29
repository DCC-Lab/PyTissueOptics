from ..geometry import primitives, Vector
from .cylinder import Cylinder


class Circle(Cylinder):
    def __init__(self, radius: float, orientation: Vector = Vector(0, 0, 1), position: Vector = Vector(0, 0, 0),
                u: int = 32, s: int = 1, primitive: str = primitives.DEFAULT, label="circle"):
        super().__init__(radius, length=0, u=u, v=1, s=s, position=position, label=label, primitive=primitive, smooth=False)
        self.orient(towards=orientation)

    def _computeTriangleMesh(self):
        # A circle is a cylinder of length 0 with a single (back) surface pointing towards the positive z axis before orientation.
        lateralLayers = self._computeSectionVertices(lateralSteps=[0], radialSteps=[self._s])
        backLayers = self._computeSectionVertices(lateralSteps=[0], radialSteps=list(range(self._s - 1, 0, -1)))
        backLayers.insert(0, lateralLayers[-1])
        backLayers.append([self._backCenter])
        self._vertices.append(self._backCenter)
        self._surfaces.add("surface", self._getSurfaceTriangles(backLayers))

    def _geometryParams(self) -> dict:
        return {
            "radius": self._radius,
            "position": self._position,
            "orientation": self._orientation,
            "u": self._u,
            "s": self._s
        }
