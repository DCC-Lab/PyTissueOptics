from pytissueoptics.scene.geometry import Vector, Quad, Triangle
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.solids.cuboidStacker import CuboidStacker
from pytissueoptics.scene.solids.stackResult import StackResult


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
                 primitive: str = primitives.DEFAULT, vertices=None, surfaceDict=None):
        self.shape = [a, b, c]

        if not vertices:
            vertices = [Vector(-a/2, -b/2, -c/2), Vector(a/2, -b/2, -c/2), Vector(a/2, b/2, -c/2), Vector(-a/2, b/2, -c/2),
                        Vector(-a/2, -b/2, c/2), Vector(a/2, -b/2, c/2), Vector(a/2, b/2, c/2), Vector(-a/2, b/2, c/2)]

        super().__init__(position=position, material=material, primitive=primitive,
                         vertices=vertices, surfaceDict=surfaceDict)

    def _computeTriangleMesh(self):
        V = self._vertices
        self._surfaceDict['Left'] = [Triangle(V[4], V[0], V[3]), Triangle(V[3], V[7], V[4])]
        self._surfaceDict['Right'] = [Triangle(V[1], V[5], V[6]), Triangle(V[6], V[2], V[1])]
        self._surfaceDict['Bottom'] = [Triangle(V[4], V[5], V[1]), Triangle(V[1], V[0], V[4])]
        self._surfaceDict['Top'] = [Triangle(V[3], V[2], V[6]), Triangle(V[6], V[7], V[3])]
        self._surfaceDict['Front'] = [Triangle(V[0], V[1], V[2]), Triangle(V[2], V[3], V[0])]
        self._surfaceDict['Back'] = [Triangle(V[5], V[4], V[7]), Triangle(V[7], V[6], V[5])]

    def _computeQuadMesh(self):
        V = self._vertices
        self._surfaceDict['Left'] = [Quad(V[4], V[0], V[3], V[7])]
        self._surfaceDict['Right'] = [Quad(V[1], V[5], V[6], V[2])]
        self._surfaceDict['Bottom'] = [Quad(V[4], V[5], V[1], V[0])]
        self._surfaceDict['Top'] = [Quad(V[3], V[2], V[6], V[7])]
        self._surfaceDict['Front'] = [Quad(V[0], V[1], V[2], V[3])]
        self._surfaceDict['Back'] = [Quad(V[5], V[4], V[7], V[6])]

    def stack(self, other: 'Cuboid', onSurface: str = 'Top') -> 'Cuboid':
        """
        Basic implementation for stacking cuboids along an axis.

        For example, stacking on 'Top' will move the other cuboid on top of the this cuboid. They will now share
         the same mesh at the interface and inside/outside materials at the interface will be properly defined.
         This will return a new cuboid that represents the stack, with a new 'Interface<i>' surface group.

        Limitations:
            - Requires cuboids with the same shape except along the stack axis.
            - Cannot stack another stack along its stacked axis (ill-defined interface material).
            - Expected behavior not guaranteed for pre-rotated cuboids.
        """
        stacker = CuboidStacker()
        stackResult = stacker.stack(onCuboid=self, otherCuboid=other, onSurface=onSurface)
        return Cuboid._fromStackResult(stackResult)

    @classmethod
    def _fromStackResult(cls, stackResult: StackResult) -> 'Cuboid':
        # subtracting stackCentroid from all vertices because solid creation will translate back to position.
        for vertex in stackResult.vertices:
            vertex.subtract(stackResult.position)

        stackSurfaces = {}
        stackSurfaces.update(stackResult.surfaces)
        stackSurfaces.update(stackResult.interfaces)

        return Cuboid(*stackResult.shape, position=stackResult.position, vertices=stackResult.vertices,
                      surfaceDict=stackSurfaces, material=None, primitive=stackResult.primitive)


if __name__ == "__main__":
    from pytissueoptics.scene.viewer.mayavi import MayaviViewer

    cuboid1 = Cuboid(5, 1, 4, position=Vector(4, 0.5, 0))
    cuboid2 = Cuboid(5, 2, 4, position=Vector(4, 1, -6))
    # cuboid3 = Cuboid(5, 2, 4, position=Vector(4, 1, -6))
    # cuboidStack = cuboid1.stack(cuboid2).stack(cuboid3, onSurface='Top')

    cuboid3 = Cuboid(2, 3, 4, position=Vector(-2, 1.5, -3))
    cuboidStack = cuboid1.stack(cuboid2).stack(cuboid3, onSurface='Right')

    viewer = MayaviViewer()
    # viewer.add(cuboid1, representation="wireframe", lineWidth=2)
    # viewer.add(cuboid2, representation="wireframe", lineWidth=2)
    # viewer.add(cuboid3, representation="wireframe", lineWidth=2)
    viewer.add(cuboidStack, representation="wireframe", lineWidth=2)
    viewer.show()
