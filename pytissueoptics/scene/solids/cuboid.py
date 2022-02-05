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
        Basic implementation for stacking cuboids along an axis. Currently requires them to have
         the same dimensions except along the stack axis.

        For example, stacking on 'Top' will move the other cuboid on top of the this cuboid. They will now share
         the same mesh at the interface and inside/outside materials at the interface will be properly handled.

        # fixme: Currently, this will yield unexpected behavior if used on previously rotated cuboids.
        """
        assert onSurface in self._surfaceDict.keys(), f"Available surfaces to stack on are: {self._surfaceDict.keys()}"
        assert "Interface" not in other._surfaceDict.keys(), "Stacking of a cuboid stack is not yet implemented."

        surfacePairs = [('Left', 'Right'), ('Bottom', 'Top'), ('Front', 'Back')]
        axis = max(axis if onSurface in surfacePair else -1 for axis, surfacePair in enumerate(surfacePairs))
        assert self.shape[(axis + 1) % 3] == other.shape[(axis + 1) % 3] and \
               self.shape[(axis + 2) % 3] == other.shape[(axis + 2) % 3], \
               f"Stacking of mismatched surfaces is not supported."

        relativePosition = [0, 0, 0]
        relativePosition[axis] = self.shape[axis]/2 + other.shape[axis]/2
        relativePosition = Vector(*relativePosition)
        other.translateTo(self.position + relativePosition)

        self._setOutsideMaterial(other._material, faceKey=onSurface)

        onSurfaceIndex = surfacePairs[axis].index(onSurface)
        oppositeSurface = surfacePairs[axis][(onSurfaceIndex + 1) % 2]

        # todo (?): remove duplicate vertices
        # there are still twice the required number of vertices at the interface
        # we could replace other.bottom vertices with self.top vertices,
        # but this process is quite involved:
        #
        # duplicateVertices = []
        # for surface in other._surfaceDict[oppositeSurface]:
        #     for vertex in surface.vertices:
        #         if vertex not in duplicateVertices:
        #             duplicateVertices.append(vertex)
        # sharedVertices = []
        # ... fill *in-order* with self.vertices with same coordinate as duplicateVertices
        # Replace other.vertices(at duplicateVertices indexes) with self.vertices(at sharedVerticesIndexes).
        # Call other.computeMesh to create proper side surfaceDict with new shared vertices reference.
        # Then we can lose reference to these duplicate surfaceDict:

        other._surfaceDict[oppositeSurface] = self._surfaceDict[onSurface]

        # Define new stack as a Cuboid
        stackCentroid = self.position + relativePosition / 2
        stackShape = self.shape.copy()
        stackShape[axis] += other.shape[axis]

        stackVertices = self._vertices
        newVertices = [vertex for vertex in other._vertices if vertex not in self._vertices]
        stackVertices.extend(newVertices)
        # subtracting stackCentroid from all vertices because solid creation will translate back to stack centroid.
        for vertex in stackVertices:
            vertex.subtract(stackCentroid)

        interfaceKeys = [key for key in self._surfaceDict.keys() if "Interface" in key]
        interfaceIndex = len(interfaceKeys)
        stackSurfaces = {onSurface: other._surfaceDict[onSurface],
                         oppositeSurface: self._surfaceDict[oppositeSurface],
                         f'Interface{interfaceIndex}': self._surfaceDict[onSurface]}
        for interfaceKey in interfaceKeys:
            stackSurfaces[interfaceKey] = self._surfaceDict[interfaceKey]
        surfaceKeysLeft = surfacePairs[(axis + 1) % 3] + surfacePairs[(axis + 2) % 3]
        for surfaceKey in surfaceKeysLeft:
            stackSurfaces[surfaceKey] = self._surfaceDict[surfaceKey] + other._surfaceDict[surfaceKey]

        # fixme: A Cuboid stack can stack other Cuboids, but not the other way around because:
        #  - currently ignores interfaces in the other cuboid
        #  - we pass None to material to skip insideMaterial reset, but that means undefined material for the stack
        # + fixme: there's a small positioning error on 3rd stack when not same axis
        # todo: refactor

        return Cuboid(*stackShape, position=stackCentroid, vertices=stackVertices, surfaceDict=stackSurfaces,
                      material=None, primitive=self._primitive)
