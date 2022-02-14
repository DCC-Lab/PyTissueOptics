from math import cos, sin, acos, atan, pi, sqrt

from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.solids import Sphere
from pytissueoptics.scene.materials import Material


class Ellipsoid(Sphere):
    """
        The Ellipsoid is based on the 3D sphere. Yes, it should be the opposite as this will lead to
        differences in mesh vertex density at the extremities of very oblong ellipsoid. However, it is easier
        to code it this way and it is fine for now. Also, the computation required is much easier just for the sphere,
        Which is in part why the Sphere isn't a child of the Ellipsoid.

        We take the unit sphere, then calculate the theta, phi position of each vertex (with ISO mathematical
        convention). Then we apply the ellipsoid formula in the spherical coordinate to isolate the component R.
        We then calculate the difference the ellipsoid would with the unit sphere for this theta,phi and
        then .add() or .subtract() the corresponding vector.
    """

    def __init__(self, a: float = 1, b: float = 1, c: float = 1, order: int = 3,
                 position: Vector = Vector(0, 0, 0), material: Material = None,
                 primitive: str = primitives.DEFAULT):

        self._a = a
        self._b = b
        self._c = c

        super().__init__(radius=1, order=order, position=position, material=material, primitive=primitive)

    @property
    def radius(self):
        return None

    def _setVerticesPositionsFromCenter(self):
        """
        The Ellipsoid parametric equation goes as: x^2/a^2 + y^2/b^2 + z^2/c^2 =1
        A Sphere is just an ellipsoid with a = b = c.
        Bringing (x, y, z) -> (theta, phi, r) we can simply take the unit sphere and stretch it,
        since the equation becomes as follow:

        r^2.cos^2(theta).sin^2(phi)/a^2 + r^2.sin^2(theta).sin^2(phi)/b^2  + r^2.cos^2(phi)/c^2 = 1
        """
        for vertex in self._vertices:
            vertex.normalize()
            r = self._radiusTowards(vertex)
            distanceFromUnitSphere = (r - 1.0)
            vertex.add(vertex * distanceFromUnitSphere)

    @staticmethod
    def _findThetaPhi(vertex: 'Vector'):
        phi = acos(vertex.z / (vertex.x ** 2 + vertex.y ** 2 + vertex.z ** 2))
        theta = 0
        if vertex.x == 0.0:
            if vertex.y > 0.0:
                theta = pi / 2

            elif vertex.y < 0.0:
                theta = -pi / 2

        elif vertex.x > 0.0:
            theta = atan(vertex.y / vertex.x)

        elif vertex.x < 0.0:
            if vertex.y >= 0.0:
                theta = atan(vertex.y / vertex.x) + pi

            elif vertex.y < 0.0:
                theta = atan(vertex.y / vertex.x) - pi

        return theta, phi
    
    def _radiusTowards(self, vertex):
        theta, phi = self._findThetaPhi(vertex)
        return sqrt(1 / ((cos(theta) ** 2 * sin(phi) ** 2) / self._a ** 2 + (
                sin(theta) ** 2 * sin(phi) ** 2) / self._b ** 2 + cos(phi) ** 2 / self._c ** 2))

    def _computeQuadMesh(self):
        raise NotImplementedError

    def contains(self, *vertices: Vector) -> bool:
        """ Only returns true if all vertices are inside the minimum radius of the sphere
        (more restrictive with low order spheres). """
        for vertex in vertices:
            relativeVertex = vertex - self.position
            if relativeVertex.getNorm() == 0:
                continue
            minRadius = self._getMinimumRadiusTowards(relativeVertex)
            if relativeVertex.getNorm() >= minRadius:
                return False
        return True

    def _getMinimumRadius(self):
        # fixme: LSP violation because ellipsoid is a child of sphere
        #  and not the other way around...
        raise NotImplementedError

    def _getMinimumRadiusTowards(self, vertex) -> float:
        return (1 - self._getRadiusError()) * self._radiusTowards(vertex)
