from pytissueoptics.scene.geometry import Vector


class Ray:
    def __init__(self, origin: Vector, direction: Vector, length: float = None):
        self._origin = origin
        self._direction = direction
        self._direction.normalize()
        self._length = length

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    @property
    def length(self):
        return self._length
