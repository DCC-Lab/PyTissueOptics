from pytissueoptics.scene.geometry import Vector


class MayaviNormals:
    def __init__(self):
        self._x = []
        self._y = []
        self._z = []

        self._u = []
        self._v = []
        self._w = []

    def add(self, position: Vector, direction: Vector):
        x, y, z = position.array
        self._x.append(x)
        self._y.append(y)
        self._z.append(z)

        u, v, w = direction.array
        self._u.append(u)
        self._v.append(v)
        self._w.append(w)

    @property
    def components(self) -> tuple:
        return self._x, self._y, self._z, self._u, self._v, self._w
