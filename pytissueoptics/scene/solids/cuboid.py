from pytissueoptics.scene.geometry import Vector, Quad, Triangle
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid


class Cuboid(Solid):
    """
        Also known as the Right Rectangular Prism, the Cuboid is defined by its
        width (a, b, c) in each axis (x, y, z) respectively.

        The position refers to the vector from global origin to its centroid.
        The generated mesh will be divided into the following subgroups:
        Left (-x), Right (+x), Bottom (-y), Top (+y), Front (-z), Back (+z).
    """

    def __init__(self, a: float, b: float, c: float,
                 position: Vector = Vector(0, 0, 0), material: Material = Material(),
                 primitive: str = primitives.DEFAULT):
        self.shape = (a, b, c)

        vertices = [Vector(-a/2, -b/2, -c/2), Vector(a/2, -b/2, -c/2), Vector(a/2, b/2, -c/2), Vector(-a/2, b/2, -c/2),
                    Vector(-a/2, -b/2, c/2), Vector(a/2, -b/2, c/2), Vector(a/2, b/2, c/2), Vector(-a/2, b/2, c/2)]

        super().__init__(position=position, material=material, vertices=vertices, primitive=primitive)

    def _computeTriangleMesh(self):
        V = self._vertices
        self._surfaces['Left'] = [Triangle(V[4], V[0], V[3]), Triangle(V[3], V[7], V[4])]
        self._surfaces['Right'] = [Triangle(V[1], V[5], V[6]), Triangle(V[6], V[2], V[1])]
        self._surfaces['Bottom'] = [Triangle(V[4], V[5], V[1]), Triangle(V[1], V[0], V[4])]
        self._surfaces['Top'] = [Triangle(V[3], V[2], V[6]), Triangle(V[6], V[7], V[3])]
        self._surfaces['Front'] = [Triangle(V[0], V[1], V[2]), Triangle(V[2], V[3], V[0])]
        self._surfaces['Back'] = [Triangle(V[5], V[4], V[7]), Triangle(V[7], V[6], V[5])]

    def _computeQuadMesh(self):
        V = self._vertices
        self._surfaces['Left'] = [Quad(V[4], V[0], V[3], V[7])]
        self._surfaces['Right'] = [Quad(V[1], V[5], V[6], V[2])]
        self._surfaces['Bottom'] = [Quad(V[4], V[5], V[1], V[0])]
        self._surfaces['Top'] = [Quad(V[3], V[2], V[6], V[7])]
        self._surfaces['Front'] = [Quad(V[0], V[1], V[2], V[3])]
        self._surfaces['Back'] = [Quad(V[5], V[4], V[7], V[6])]

    def stack(self, other: 'Cuboid', onSurface: str = 'Top'):
        """
        Basic implementation for stacking cuboids along an axis. They need to have the same dimensions except
            for the stack axis.

        For example, stacking on 'Top' will move the other cuboid on top of the this cuboid. They will now share
            the same mesh at the interface and inside/outside materials at the interface will be properly handled.

        # fixme: Currently, this will yield unexpected behavior if used on previously rotated cuboids.
        """
        assert onSurface in self._surfaces.keys(), f"Available surfaces to stack on are: {self._surfaces.keys()}"

        surfacePairs = [('Left', 'Right'), ('Bottom', 'Top'), ('Front', 'Back')]
        axis = max(axis if onSurface in surfacePair else -1 for axis, surfacePair in enumerate(surfacePairs))
        assert self.shape[(axis + 1) % 3] == other.shape[(axis + 1) % 3] and \
               self.shape[(axis + 2) % 3] == other.shape[(axis + 2) % 3], \
               f"Stacking of mismatched surfaces is not supported."

        relativePosition = [0, 0, 0]
        relativePosition[axis] = self.shape[axis]/2 + other.shape[axis]/2
        relativePosition = Vector(*relativePosition)
        other.translateTo(self.position + relativePosition)
