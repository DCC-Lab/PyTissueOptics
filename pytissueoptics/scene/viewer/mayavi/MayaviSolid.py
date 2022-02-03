from pytissueoptics.scene.solids import Solid


class MayaviSolid:
    def __init__(self, solid: 'Solid'):
        self._solid = solid
        self._x = []
        self._y = []
        self._z = []
        self._triangleIndicesTuples = []
        self._vertexHash = []

        self._translate()

    def _translate(self):
        vertexHash = []
        for vertex in self._solid.vertices:
            vertexHash.append(hash(vertex))
            self._x.append(vertex.x)
            self._x.append(vertex.y)
            self._x.append(vertex.z)

        for surface in self._solid.surfaces:
            surfaceIndices = []
            for vertex in surface.vertices:
                index = self._vertexHash.index(hash(vertex))
                surfaceIndices.append(index)
            self._triangleIndicesTuples.append(tuple(surfaceIndices))

    @property
    def meshComponents(self):
        return self._x, self._y, self._z, self._triangleIndicesTuples

