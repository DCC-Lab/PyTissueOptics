from pytissueoptics.scene.solids import Solid


class MayaviSolid:
    def __init__(self, solid: 'Solid'):
        self._solid = solid
        self._x = []
        self._y = []
        self._z = []
        self._triangleIndicesTuples = []

        self._translate()

    def _translate(self):
        self._separateXYZ()
        self._findTriangleIndices()

    def _separateXYZ(self):
        for vertex in self._solid.vertices:
            self._x.append(vertex.x)
            self._y.append(vertex.y)
            self._z.append(vertex.z)

    def _findTriangleIndices(self):
        for surface in self._solid.surfaces:
            surfaceIndices = []
            for vertex in surface.vertices:
                index = self._solid.vertices.index(vertex)
                surfaceIndices.append(index)
            self._triangleIndicesTuples.append(tuple(surfaceIndices))

    @property
    def meshComponents(self):
        return self._x, self._y, self._z, self._triangleIndicesTuples

