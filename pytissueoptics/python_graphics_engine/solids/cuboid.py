from pytissueoptics.python_graphics_engine.geometry import Vector, Quad, Triangle
from pytissueoptics.python_graphics_engine.geometry import primitives
from pytissueoptics.python_graphics_engine.materials import Material
from pytissueoptics.python_graphics_engine.solids import Solid


class Cuboid(Solid):
    """
        Also known as the Right Rectangular Prism, the Cuboid is defined by its
        width (a, b, c) in each axis (x, y, z) respectively.

        The position refers to the vector from global origin to its centroid.
        The generated mesh will be divided into the following subgroups:
        Front (-z), Back (+z), Left (-x), Right (+x), Bottom (-y), Top (+y).
    """

    def __init__(self, a: float, b: float, c: float,
                 position: Vector = Vector(0, 0, 0), material: Material = Material(),
                 primitive: str = primitives.DEFAULT):

        vertices = [Vector(-a/2, -b/2, -c/2), Vector(a/2, -b/2, -c/2), Vector(a/2, b/2, -c/2), Vector(-a/2, b/2, -c/2),
                    Vector(-a/2, -b/2, c/2), Vector(a/2, -b/2, c/2), Vector(a/2, b/2, c/2), Vector(-a/2, b/2, c/2)]
        surfaces = {'Front': [], 'Back': [], 'Left': [], 'Right': [], 'Top': [], 'Bottom': []}

        super().__init__(position=position, material=material, vertices=vertices, surfaces=surfaces,
                         primitive=primitive)

    def _computeTriangleMesh(self):
        V = self._vertices
        self._surfaces['Front'] = [Triangle(V[0], V[1], V[2]), Triangle(V[2], V[3], V[0])]
        self._surfaces['Back'] = [Triangle(V[5], V[4], V[7]), Triangle(V[7], V[6], V[5])]
        self._surfaces['Left'] = [Triangle(V[4], V[0], V[3]), Triangle(V[3], V[7], V[4])]
        self._surfaces['Right'] = [Triangle(V[1], V[5], V[6]), Triangle(V[6], V[2], V[1])]
        self._surfaces['Top'] = [Triangle(V[3], V[2], V[6]), Triangle(V[6], V[7], V[3])]
        self._surfaces['Bottom'] = [Triangle(V[4], V[5], V[1]), Triangle(V[1], V[0], V[4])]

    def _computeQuadMesh(self):
        V = self._vertices
        self._surfaces['Front'] = [Quad(V[0], V[1], V[2], V[3])]
        self._surfaces['Back'] = [Quad(V[5], V[4], V[7], V[6])]
        self._surfaces['Left'] = [Quad(V[4], V[0], V[3], V[7])]
        self._surfaces['Right'] = [Quad(V[1], V[5], V[6], V[2])]
        self._surfaces['Top'] = [Quad(V[3], V[2], V[6], V[7])]
        self._surfaces['Bottom'] = [Quad(V[4], V[5], V[1], V[0])]
