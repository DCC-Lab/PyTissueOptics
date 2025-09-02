from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.solids import Ellipsoid


class Sphere(Ellipsoid):
    """
    The Sphere is the 3D analog to the circle. Meshing a sphere requires an infinite number of vertices.
    The position refers to the vector from global origin to its centroid.
    The radius of the sphere will determine the outermost distance from its centroid.

    This class offers two possible methods to generate the sphere mesh.
    - With Quads: Specify the number of separation lines on the vertical axis and the horizontal axis of the sphere.
    - With Triangle: Specify the order of splitting. This will generate what is known as an IcoSphere.
    """

    def __init__(
        self,
        radius: float = 1.0,
        order: int = 3,
        position: Vector = Vector(0, 0, 0),
        material=None,
        label: str = "sphere",
        primitive: str = primitives.DEFAULT,
        smooth: bool = True,
    ):
        self._radius = radius

        super().__init__(
            a=radius,
            b=radius,
            c=radius,
            order=order,
            position=position,
            material=material,
            label=label,
            primitive=primitive,
            smooth=smooth,
        )

    @property
    def radius(self):
        return self._radius

    def _computeQuadMesh(self):
        raise NotImplementedError

    def contains(self, *vertices: Vector) -> bool:
        """Only returns true if all vertices are inside the minimum radius of the sphere
        (more restrictive with low order spheres)."""
        minRadius = self._getMinimumRadius()
        for vertex in vertices:
            relativeVertex = vertex - self.position
            if relativeVertex.getNorm() >= minRadius:
                return False
        return True

    def _getMinimumRadius(self) -> float:
        return (1 - self._getRadiusError()) * self._radius

    def _radiusTowards(self, vertex) -> float:
        return self.radius

    def _geometryParams(self) -> dict:
        return {
            "radius": self._radius,
            "order": self._order,
        }
