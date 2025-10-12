from ..geometry import Quad, Triangle, Vector, Vertex, primitives
from .solid import Solid


class Rectangle(Solid):
    def __init__(
        self,
        a: float,
        b: float,
        orientation=Vector(0, 0, 1),
        position=Vector(0, 0, 0),
        primitive: str = primitives.DEFAULT,
        label="rectangle",
    ):
        self._a = a
        self._b = b

        vertices = [  # Mesh points towards (0, 0, 1) by default, as expected by the orient() method.
            Vertex(-a / 2, -b / 2, 0),
            Vertex(a / 2, -b / 2, 0),
            Vertex(a / 2, b / 2, 0),
            Vertex(-a / 2, b / 2, 0),
        ]

        super().__init__(vertices, position, label=label, primitive=primitive)
        self.orient(towards=orientation)

    def _computeTriangleMesh(self):
        V = self._vertices
        self._surfaces.add("surface", [Triangle(V[0], V[1], V[2]), Triangle(V[2], V[3], V[0])])

    def _computeQuadMesh(self):
        V = self._vertices
        self._surfaces.add("surface", [Quad(V[0], V[1], V[2], V[3])])

    def _geometryParams(self) -> dict:
        return {"a": self._a, "b": self._b, "position": self._position, "orientation": self._orientation}

    @property
    def isFlat(self) -> bool:
        return True
